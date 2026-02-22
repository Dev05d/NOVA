package com.example.impairedapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.lifecycleScope
import com.example.impairedapp.ui.theme.ImpairedAppTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume

// --- DATA CLASSES & INTERFACES (UNCHANGED) ---

data class ServerDetectionResponse(
    val objects: List<List<Any>>
)

interface PythonServerApiService {
    @Multipart
    @POST("send")
    suspend fun detectObjects(@Part file: MultipartBody.Part): Response<ServerDetectionResponse>
}

class MainActivity : ComponentActivity() {

    private val USE_MOCK_MODE = false
    private val PYTHON_SERVER_URL_MATEO = "http://172.31.81.122:8000/"
    private val PYTHON_SERVER_URL = "http://172.31.177.224:8000/"

    private var nativeTts: TextToSpeech? = null
    private lateinit var cameraExecutor: ExecutorService
    private var mediaPlayer: MediaPlayer? = null

    private val httpClient = OkHttpClient()

    private var lastSpokenTime = 0L
    private val SPEAK_COOLDOWN_MS = 1L

    @Volatile private var isSpeaking = false
    @Volatile private var isProcessingFrame = false

    // FIX FOR "UNEXPECTED END OF STREAM":
    // The interceptor forces "Connection: close" safely, avoiding OkHttp connection pool mismatches with Python servers.
    private val serverHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .addInterceptor { chain ->
            val request = chain.request().newBuilder()
                .header("Connection", "close")
                .build()
            chain.proceed(request)
        }
        .build()

    private val pythonServerService: PythonServerApiService by lazy {
        Retrofit.Builder()
            .baseUrl(PYTHON_SERVER_URL)
            .client(serverHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(PythonServerApiService::class.java)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (USE_MOCK_MODE) {
            nativeTts = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) nativeTts?.language = Locale.US
            }
        }

        setContent {
            ImpairedAppTheme {
                AppContent()
            }
        }
    }

    @Composable
    private fun AppContent() {
        var hasCameraPermission by remember {
            mutableStateOf(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
        }
        var hasAttemptedRequest by remember { mutableStateOf(false) }

        val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            hasCameraPermission = granted
        }

        val lifecycleOwner = LocalLifecycleOwner.current

        DisposableEffect(lifecycleOwner) {
            val observer = LifecycleEventObserver { _, event ->
                if (event == Lifecycle.Event.ON_RESUME && !hasAttemptedRequest && !hasCameraPermission) {
                    hasAttemptedRequest = true
                    launcher.launch(Manifest.permission.CAMERA)
                }
            }
            lifecycleOwner.lifecycle.addObserver(observer)
            onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
        }

        Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
            if (hasCameraPermission) {
                CameraScreen(
                    modifier = Modifier.padding(innerPadding),
                    cameraExecutor = cameraExecutor,
                    isProcessingFrame = { isProcessingFrame },
                    onImageCaptured = { bitmap -> processImageOnServer(bitmap) }
                )
            } else {
                PermissionScreen(launcher)
            }
        }
    }

    @Composable
    private fun PermissionScreen(launcher: androidx.activity.result.ActivityResultLauncher<String>) {
        val context = LocalContext.current
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(text = "Camera permission is required to see objects.")
                Spacer(modifier = Modifier.height(16.dp))
                Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) {
                    Text("Request Permission")
                }
                Spacer(modifier = Modifier.height(8.dp))
                Button(onClick = {
                    val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                        data = Uri.fromParts("package", context.packageName, null)
                    }
                    context.startActivity(intent)
                }) {
                    Text("Open Settings")
                }
            }
        }
    }

    private fun processImageOnServer(bitmap: Bitmap) {
        if (USE_MOCK_MODE) {
            handleSpeaking(listOf(Pair("mock person", 1.5), Pair("mock car", 3.6)))
            return
        }

        if (isProcessingFrame) {
            bitmap.recycle() // Safely drop the frame
            return
        }

        isProcessingFrame = true

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Scale and Compress
                val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 640, 480, true)
                val stream = ByteArrayOutputStream()
                scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
                val jpegBytes = stream.toByteArray()

                // Cleanup bitmaps immediately to prevent memory leaks
                scaledBitmap.recycle()
                bitmap.recycle()

                val requestBody = jpegBytes.toRequestBody("image/jpeg".toMediaTypeOrNull())
                val multipartBody = MultipartBody.Part.createFormData("file", "image.jpg", requestBody)

                val response = pythonServerService.detectObjects(multipartBody)

                if (response.isSuccessful && response.body() != null) {
                    val detectedTuples = response.body()!!.objects.mapNotNull { item ->
                        if (item.size >= 2) {
                            val name = item[0] as? String
                            val distance = (item[1] as? Number)?.toDouble()
                            if (name != null && distance != null) Pair(name, distance) else null
                        } else null
                    }

                    if (detectedTuples.isNotEmpty()) {
                        handleSpeaking(detectedTuples)
                    }
                } else {
                    Log.e("ServerAPI", "Server error: ${response.code()}")
                }

            } catch (e: Exception) {
                Log.e("ServerAPI", "Network request failed: ${e.message}")
            } finally {
                // Added a delay here to throttle how fast we send frames to the server.
                // Using NonCancellable ensures the flag is ALWAYS reset even if the lifecycle scope cancels during the delay.
                withContext(kotlinx.coroutines.NonCancellable) {
                   // 2-second cooldown delay between server requests (adjust this number as needed)
                    kotlinx.coroutines.delay(2000)
                    isProcessingFrame = false
                }
            }
        }
    }

    private fun handleSpeaking(detectedObjects: List<Pair<String, Double>>) {
        // 1. Check if we are currently speaking
        if (!isSpeaking) {
            // 2. Lock it IMMEDIATELY on the current thread before launching the coroutine
            isSpeaking = true

            val textToSpeak = detectedObjects.distinct().joinToString(". ") { (name, distance) ->
                val formattedDistance = String.format(Locale.US, "%.1f", distance)
                "$name at $formattedDistance meters detected"
            }

            lifecycleScope.launch(Dispatchers.IO) {
                try {
                    Log.d("SPEECH_TRIGGER", "Speaking batch: $textToSpeak")
                    // UNCOMMENT TO ENABLE AUDIO
                    speakAndWait(textToSpeak)
                } finally {
                    // 3. Always unlock when finished, even if it crashes
                    isSpeaking = false
                }
            }
        }
    }

    private suspend fun speakAndWait(text: String) {
        if (USE_MOCK_MODE) {
            nativeTts?.speak(text, TextToSpeech.QUEUE_ADD, null, null)
            kotlinx.coroutines.delay(1)
            return
        }

        val apiKey = "sk_df317a2498125ae096d0b943c48399c0c11b0d5c3ee6e624"
        val voiceId = "iP95p4xoKVk53GoZ742B"
        val jsonBody = """{"text": "$text", "model_id": "eleven_multilingual_v2"}"""

        val request = Request.Builder()
            .url("https://api.elevenlabs.io/v1/text-to-speech/$voiceId")
            .addHeader("xi-api-key", apiKey)
            .post(jsonBody.toRequestBody("application/json".toMediaTypeOrNull()))
            .build()

        try {
            // Execute the network request. This is where an IOException will be thrown
            // if there is no internet or the connection times out.
            val response = httpClient.newCall(request).execute()

            // Check if the server responded with a success code (e.g., 200 OK)
            if (response.isSuccessful) {
                response.body?.bytes()?.let { audioBytes ->
                    val outputFile = File(cacheDir, "output.mp3")
                    outputFile.writeBytes(audioBytes)

                    // Only attempt to play if we successfully saved the bytes
                    withContext(Dispatchers.Main) { playMp3AndWait(outputFile) }
                }
            } else {
                // The request reached ElevenLabs, but they returned an error
                // (e.g., 401 Unauthorized, 429 Rate Limit exceeded)
                Log.e("ElevenLabs", "API Error: ${response.code} - ${response.message}")
            }
        } catch (e: java.io.IOException) {
            // Handle physical network failures here (no Wi-Fi, DNS failure, etc.)
            Log.e("ElevenLabs", "Network Error: Could not reach ElevenLabs", e)
        } catch (e: Exception) {
            // A safety net for any other unforeseen errors (e.g., file system issues)
            Log.e("ElevenLabs", "Unexpected error during TTS generation", e)
        }
    }

    private suspend fun playMp3AndWait(file: File) = withContext(Dispatchers.Main) {
        suspendCancellableCoroutine { continuation ->
            try {
                mediaPlayer?.release()
                mediaPlayer = MediaPlayer().apply {
                    setDataSource(file.absolutePath)
                    setOnCompletionListener {
                        if (continuation.isActive) continuation.resume(Unit)
                    }
                    setOnErrorListener { _, _, _ ->
                        if (continuation.isActive) continuation.resume(Unit)
                        true
                    }
                    prepareAsync()
                    setOnPreparedListener { start() }
                }
            } catch (e: Exception) {
                Log.e("AudioSystem", "Playback exception", e)
                if (continuation.isActive) continuation.resume(Unit)
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        nativeTts?.stop()
        nativeTts?.shutdown()
        mediaPlayer?.release()
    }
}

@Composable
fun CameraScreen(
    modifier: Modifier = Modifier,
    cameraExecutor: ExecutorService,
    isProcessingFrame: () -> Boolean,
    onImageCaptured: (Bitmap) -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current

    AndroidView(
        modifier = modifier.fillMaxSize(),
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    // STRATEGY_KEEP_ONLY_LATEST natively handles frame drops, no need for manual timestamps
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor) { imageProxy ->
                                if (!isProcessingFrame()) {
                                    onImageCaptured(imageProxy.toBitmap())
                                }
                                imageProxy.close() // ALWAYS close the proxy immediately
                            }
                        }

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalysis
                    )
                } catch (e: Exception) {
                    Log.e("CameraScreen", "Critical Camera failure", e)
                }
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        }
    )
}













