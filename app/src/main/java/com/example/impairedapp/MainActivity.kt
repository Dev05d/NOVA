package com.example.impairedapp

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
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
import kotlinx.coroutines.delay
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
import androidx.core.graphics.scale
import kotlin.coroutines.resume

// --- DATA CLASSES & INTERFACES ---

// 1. Python Server Data & Interface
data class ServerDetectionResponse(val objects: List<String>)

interface PythonServerApiService {
    @Multipart
    @POST("send") // Change this to your server's endpoint path
    suspend fun detectObjects(@Part image: MultipartBody.Part): Response<ServerDetectionResponse>
}


class MainActivity : ComponentActivity() {

    // --- CONFIGURATION ---
    private val USE_MOCK_MODE = false
    // Replace with your actual computer/server IP address (e.g., "http://192.168.1.100:5000/")
    private val PYTHON_SERVER_URL = "http://172.31.177.224:8000/"

    // Native Android TTS for Mock Mode
    private var nativeTts: TextToSpeech? = null
    private lateinit var cameraExecutor: ExecutorService

    // Audio Playback
    private var mediaPlayer: MediaPlayer? = null
    private val httpClient = OkHttpClient()

    // Audio Debounce state
    private var lastSpokenTime = 0L
    private val SPEAK_COOLDOWN_MS = 4000L // Wait 4 seconds before speaking again

    // Prevent overlapping speech loops
    @Volatile
    private var isSpeaking = false

    // Prevent network queue buildup if network is slower than 20fps
    // @Volatile ensures thread safety since multiple background threads touch this now
    @Volatile
    private var isProcessingFrame = false

    private val pythonServerService: PythonServerApiService by lazy {
        Retrofit.Builder()
            .baseUrl(PYTHON_SERVER_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(PythonServerApiService::class.java)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (USE_MOCK_MODE) {
            nativeTts = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    nativeTts?.language = Locale.US
                }
            }
            Log.d("MOCK_MODE", "Running in Mock Mode! Network calls are disabled.")
        }

        setContent {
            ImpairedAppTheme {
                var hasCameraPermission by remember {
                    mutableStateOf(
                        ContextCompat.checkSelfPermission(
                            this@MainActivity,
                            Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                    )
                }

                // Track if we've already tried automatically asking so it doesn't spam the user
                var hasAttemptedRequest by remember { mutableStateOf(false) }

                val launcher = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.RequestPermission(),
                    onResult = { granted -> hasCameraPermission = granted }
                )

                val lifecycleOwner = LocalLifecycleOwner.current

                // Wait until the App is fully open and visible on the screen to automatically ask for permission
                DisposableEffect(lifecycleOwner) {
                    val observer = LifecycleEventObserver { _, event ->
                        if (event == Lifecycle.Event.ON_RESUME && !hasAttemptedRequest && !hasCameraPermission) {
                            hasAttemptedRequest = true
                            launcher.launch(Manifest.permission.CAMERA)
                        }
                    }
                    lifecycleOwner.lifecycle.addObserver(observer)
                    onDispose {
                        lifecycleOwner.lifecycle.removeObserver(observer)
                    }
                }

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    if (hasCameraPermission) {
                        CameraScreen(
                            modifier = Modifier.padding(innerPadding),
                            cameraExecutor = cameraExecutor, // Pass the background executor to the camera
                            isProcessingFrame = { isProcessingFrame }, // Pass the boolean state down to prevent memory crashes
                            onImageCaptured = { bitmap ->
                                processImageOnServer(bitmap)
                            }
                        )
                    } else {
                        val context = LocalContext.current

                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(text = "Camera permission is required to see objects.")
                                Spacer(modifier = Modifier.height(16.dp))

                                // Standard Android Request fallback
                                Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) {
                                    Text("Request Permission Popup")
                                }

                                Spacer(modifier = Modifier.height(8.dp))

                                // Deep link to the Phone's App Settings if the popup is permanently blocked
                                Button(onClick = {
                                    val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                                        data = Uri.fromParts("package", context.packageName, null)
                                    }
                                    context.startActivity(intent)
                                }) {
                                    Text("Open Phone Settings")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun processImageOnServer(bitmap: Bitmap) {
        if (USE_MOCK_MODE) {
            handleSpeaking(listOf("mock person", "mock car"))
            return
        }

        // Double check to prevent overlapping network calls
        if (isProcessingFrame) {
            bitmap.recycle() // Free memory instantly if dropped
            return
        }
        isProcessingFrame = true

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // 1. Scale down the image to save network bandwidth (e.g. 640x480)
                val scaledBitmap = bitmap.scale(640, 480)
                bitmap.recycle() // CRITICAL: Free the massive original 12MP bitmap

                // 2. Compress the Bitmap to JPEG bytes
                val stream = ByteArrayOutputStream()
                scaledBitmap.compress(Bitmap.CompressFormat.JPEG, 80, stream)
                scaledBitmap.recycle() // CRITICAL: Free the scaled bitmap
                val jpegByteArray = stream.toByteArray()

                // 3. Prepare the Multipart request
                val requestBody = jpegByteArray.toRequestBody("image/jpeg".toMediaTypeOrNull())
                val multipartBody = MultipartBody.Part.createFormData("file", "image.jpg", requestBody)

                // 4. Send to Python Server
                val response = pythonServerService.detectObjects(multipartBody)

                if (response.isSuccessful) {
                    val serverResponse = response.body()
                    val detectedObjects = serverResponse?.objects ?: emptyList()

                    if (detectedObjects.isNotEmpty()) {
//                        Log.d("ServerAPI", "Detected objects: $detectedObjects")
                        // UNCOMMENTED: Trigger the speech handler
                        handleSpeaking(detectedObjects)

                    }
                } else {
                    Log.e("ServerAPI", "Server error: ${response.code()}")
                }

            } catch (e: Exception) {
                Log.e("ServerAPI", "Network request failed", e)
            } finally {
                // Unlock frame processing for the next available frame
                isProcessingFrame = false
            }
        }
    }

    private fun handleSpeaking(detectedObjects: List<String>) {
        val currentTime = System.currentTimeMillis()

        // Ensure cooldown has passed AND we arent currently in the middle of speaking
        if (currentTime - lastSpokenTime > SPEAK_COOLDOWN_MS && !isSpeaking) {
            lastSpokenTime = currentTime

            val uniqueObjects = detectedObjects.distinct()

            // Format for accessibility: "person detected. tree detected."
            // We join them with a period so ElevenLabs naturally pauses between each one,
            // while still only requiring ONE fast internet request.
            val textToSpeak = uniqueObjects.joinToString(". ") { "$it detected" }

            // Launch a coroutine to handle the single network request and playback
            lifecycleScope.launch(Dispatchers.IO) {
                isSpeaking = true // Lock the speech system

                Log.d("SPEECH_TRIGGER", "Speaking batch: $textToSpeak")
//                speakAndWait(textToSpeak)

                isSpeaking = false // Unlock the speech system once the audio finishes playing
            }
        }
    }

    // Notice 'suspend' keyword: This allows the Coroutine to pause here until finished.
    private suspend fun speakAndWait(text: String) {
        if (USE_MOCK_MODE) {
            Log.d("ElevenLabsMock", "Terminal Output: Would have spoken -> '$text'")
            // QUEUE_ADD ensures native TTS doesn't cut itself off
            nativeTts?.speak(text, TextToSpeech.QUEUE_ADD, null, null)
            delay(1000) // Simulate waiting for native speech
            return
        }

        try {
            val apiKey = "sk_df317a2498125ae096d0b943c48399c0c11b0d5c3ee6e624"
            val voiceId = "iP95p4xoKVk53GoZ742B"
            val url = "https://api.elevenlabs.io/v1/text-to-speech/$voiceId"

            val jsonBody = """{"text": "$text", "model_id": "eleven_multilingual_v2"}"""
            val requestBody = jsonBody.toRequestBody("application/json".toMediaTypeOrNull())

            val request = Request.Builder()
                .url(url)
                .addHeader("xi-api-key", apiKey)
                .addHeader("Content-Type", "application/json")
                .post(requestBody)
                .build()

            // Making just ONE network call for the entire sentence is much faster!
            val response = httpClient.newCall(request).execute()

            if (response.isSuccessful) {
                val audioBytes = response.body?.bytes()
                if (audioBytes != null) {
                    val outputFile = File(cacheDir, "output.mp3")
                    outputFile.writeBytes(audioBytes)
                    Log.d("ElevenLabs", "Audio saved as output.mp3")

                    // Switch to Main Thread to play the MP3, and WAIT for it to finish
                    withContext(Dispatchers.Main) {
                        playMp3AndWait(outputFile)
                    }
                }
            } else {
                Log.e("ElevenLabs", "Error: ${response.code} - ${response.message}")
            }
        } catch (e: Exception) {
            Log.e("ElevenLabs", "Error calling TTS", e)
        }
    }

    // A suspend function that wraps the MediaPlayer. It resumes the Coroutine only when the track finishes.
    private suspend fun playMp3AndWait(file: File) = suspendCancellableCoroutine { continuation ->
        try {
            mediaPlayer?.release()
            mediaPlayer = MediaPlayer().apply {
                setDataSource(file.absolutePath)

                // When audio finishes naturally, resume the Coroutine
                setOnCompletionListener {
                    continuation.resume(Unit)
                }

                // If there's an error, resume anyway so the App doesn't hang forever
                setOnErrorListener { _, _, _ ->
                    continuation.resume(Unit)
                    true
                }

                prepare()
                start()
            }
        } catch (e: Exception) {
            Log.e("AudioSystem", "Failed to play MP3", e)
            if (continuation.isActive) {
                continuation.resume(Unit)
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
    cameraExecutor: ExecutorService, // Accept executor as parameter
    isProcessingFrame: () -> Boolean, // Callback to check network status
    onImageCaptured: (Bitmap) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // This prevents the camera from infinitely restarting itself 20 times a second.
    var lastAnalyzedTimestamp = remember { 0L }
    val targetFps = 25
    val frameThrottleMs = 1000L / targetFps

    // Moved camera initialization into the `factory` so it only runs ONCE.
    AndroidView(
        modifier = modifier.fillMaxSize(),
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx.applicationContext)

            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    // CRITICAL HARDWARE FIX: Removed forced Resolution and Format.
                    // Forcing combinations causes many phones to crash immediately. Let CameraX decide the safest format.
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also {
                            // Use cameraExecutor background thread instead of MainExecutor!
                            it.setAnalyzer(cameraExecutor) { imageProxy ->
                                val currentTimestamp = System.currentTimeMillis()

                                // Throttle to ~20 FPS (every 50ms)
                                if (currentTimestamp - lastAnalyzedTimestamp >= frameThrottleMs) {
                                    lastAnalyzedTimestamp = currentTimestamp

                                    // CRITICAL OOM FIX: Do NOT create a Bitmap if we are currently uploading to the server!
                                    if (!isProcessingFrame()) {
                                        try {
                                            val bitmap = imageProxy.toBitmap()
                                            onImageCaptured(bitmap)
                                        } catch (e: Exception) {
                                            Log.e("CameraScreen", "Bitmap error", e)
                                        }
                                    }
                                }
                                // Always close the imageProxy to free up memory for the next frame
                                imageProxy.close()
                            }
                        }

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                } catch (e: Exception) {
                    Log.e("CameraScreen", "Critical Camera failure", e)
                }
            }, ContextCompat.getMainExecutor(ctx)) // Leave this as MainExecutor (UI binding needs Main Thread)

            previewView // Return the view
        }
    )
}