package com.example.image_tagger

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import java.util.*

class TagChooserDialog(context: Context) : Dialog(context) {

    val parent: TaggerActivity = context as TaggerActivity

    private lateinit var tagList: ListView
    private lateinit var tagMapAdapter: TagMapAdapter
    private lateinit var filterTags: EditText
    private lateinit var addNewTag: ImageButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dialog_tags)

        tagList = findViewById(R.id.tag_list)
        filterTags = findViewById(R.id.filter_tags)
        addNewTag = findViewById(R.id.add_new_tag_button)

        tagMapAdapter = TagMapAdapter(this, parent.theTags)
        tagList.adapter = tagMapAdapter

        filterTags.requestFocus()
        parent.showKeyboard()
        filterTags.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) { }

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) { }

            override fun afterTextChanged(s: Editable?) {
                with(tagList.adapter as TagMapAdapter) {
                    hideIfNoMatch(s.toString())
                    notifyDataSetChanged()
                }
            }
        })

        addNewTag.setOnClickListener {
            val label = filterTags.editableText.toString()

            if (label.isNotEmpty()) {
                it.context.hideKeyboard(it)

                val newClassId = parent.theTags.keys
                        .maxOrNull()
                        ?.let { maxOldClassId -> maxOldClassId + 1 }
                parent.addNewChip(newClassId, label)

                dismiss()
            }
        }
    }

}

class TagMapAdapter(private val listDialog: TagChooserDialog, tags: Map<Int, String?>) : BaseAdapter() {

    private val listItems = tags.toList()
        .sortedBy { (_, label) -> label?.toUpperCase(Locale.getDefault()) ?: ""}
        .map { it.first.toLong() to it.second.toString() }

    private val itemIndices = listItems.indices
        .mapIndexed { i, itemIdx -> Pair(i, itemIdx) }
        .toMap(mutableMapOf())

    override fun getCount(): Int = itemIndices.size

    override fun getItem(position: Int): String = itemIndices[position]?.let { index ->
        listItems[index].second
    } ?: ""

    override fun getItemId(position: Int): Long = itemIndices[position]?.let { index ->
        listItems[index].first
    } ?: 0L

    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {
        val cv = convertView ?: LayoutInflater.from(listDialog.context)
                .inflate(R.layout.dialog_tags_row, parent, false)

        val classId = getItemId(position).toInt()
        val label = getItem(position)
        val textViewRow = cv.findViewById<TextView>(R.id.tag_entry)
        textViewRow.text = label

        textViewRow.setOnClickListener {
            it.context.hideKeyboard(it)
            listDialog.parent.addNewChip(classId, label)
            listDialog.dismiss()
        }

        return cv
    }

    fun hideIfNoMatch(s: String) {
        itemIndices.clear()
        var i = 0
        for (itemIdx in listItems.indices) {
            with (listItems[itemIdx].second) {
                if (s.isEmpty() ||
                    this.contains(s, ignoreCase = true) ||
                    s.contains(this, ignoreCase = true)) {
                        itemIndices[i++] = itemIdx
                }
            }
        }
    }
}
