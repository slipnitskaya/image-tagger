package com.example.caption_assistant

import android.net.Uri
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity


class CaptionActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private var imageUri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_caption)

        imageView = findViewById(R.id.chosen_image)
    }

    override fun onStart() {
        super.onStart()

        imageUri = intent.extras?.get("ImageURI") as Uri?
        if (imageUri != null) {
            imageView.setImageURI(imageUri)
        }
    }
}
