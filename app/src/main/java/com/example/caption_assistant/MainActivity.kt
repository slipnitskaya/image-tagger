package com.example.caption_assistant

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import java.io.File
import java.io.FileOutputStream


class MainActivity : AppCompatActivity() {

    lateinit var uploadButton: Button
    lateinit var cameraButton: Button
    lateinit var imageView: ImageView

    private enum class Action {
        PICK_IMAGE,
        CAPTURE_IMAGE;
        operator fun invoke(): Int = ordinal
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.background)

        uploadButton = findViewById(R.id.upload_button)
        uploadButton.setOnClickListener {
            val getIntent = Intent(Intent.ACTION_GET_CONTENT)
            getIntent.type = "image/*"

            val pickIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            pickIntent.type = "image/*"

            val chooserIntent = Intent.createChooser(getIntent, "Select Image")
            chooserIntent.putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(pickIntent))

            startActivityForResult(chooserIntent, Action.PICK_IMAGE())
        }

        cameraButton = findViewById(R.id.camera_button)
        cameraButton.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(takePictureIntent, Action.CAPTURE_IMAGE())
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), Action.CAPTURE_IMAGE())
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == Action.CAPTURE_IMAGE()) {
            permissions.forEachIndexed { index, perm ->
                if (perm == Manifest.permission.CAMERA) {
                    cameraButton.isEnabled = grantResults[index] > 0
                }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (Action.values().any { it() == requestCode } && resultCode == RESULT_OK) {
            val captionIntent = Intent(baseContext, CaptionActivity::class.java)
            var imageUri: Uri? = null

            when (requestCode) {
                Action.PICK_IMAGE() -> {
                    imageUri = data?.data
                }
                Action.CAPTURE_IMAGE() -> {
                    val bitmap = data?.extras?.get("data") as Bitmap?
                    val tempDir = externalCacheDir ?: cacheDir
                    val tempFile = File.createTempFile("CaptionAssistant", "Preview", tempDir)
                    FileOutputStream(tempFile).use {
                        bitmap?.compress(Bitmap.CompressFormat.PNG, 90, it)
                    }
                    imageUri = tempFile.toUri()
                }
            }

            captionIntent.putExtra("ImageURI", imageUri)
            startActivity(captionIntent)
        }
    }
}
