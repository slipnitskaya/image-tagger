package com.example.caption_assistant

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.core.view.get
import androidx.core.view.size
import com.google.android.material.chip.Chip
import java.util.*

class TagListDialog(context: Context, private val tags: Map<Long, String>) : Dialog(context) {

    var parentActivity: CaptionActivity = context as CaptionActivity
    private lateinit var tagList: ListView
    private lateinit var filterTags: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.dialog_tags)

        tagList = findViewById(R.id.tag_list)
        filterTags = findViewById(R.id.filter_tags)
        tagList.adapter = MapAdapter(this, tags)

        filterTags.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) { }

            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) { }

            override fun afterTextChanged(s: Editable?) {
                with(tagList.adapter as MapAdapter) {
                    hideIfNoMatch(s.toString())
                    notifyDataSetChanged()
                }
            }

        })

        filterTags.requestFocus()
    }
}

class MapAdapter(private val listDialog: TagListDialog, tags: Map<Long, String>) : BaseAdapter() {

    private val context = listDialog.context
    private val listItems = tags.toList().sortedBy { (_, label) -> label.toUpperCase(Locale.getDefault()) }
    private val itemIndices = listItems.indices.mapIndexed { i, itemIdx -> Pair(i, itemIdx) }.toMap(mutableMapOf())

    override fun getCount(): Int = itemIndices.size

    override fun getItem(position: Int): String = listItems[itemIndices[position]!!].second

    override fun getItemId(position: Int): Long = listItems[itemIndices[position]!!].first

    override fun getView(position: Int, convertView: View?, parent: ViewGroup?): View {
        val cv = convertView ?: LayoutInflater.from(context)
                .inflate(R.layout.dialog_tags_row, parent, false)

        val classId = getItemId(position)
        val label = getItem(position)
        val textViewRow = cv.findViewById<TextView>(R.id.tag_entry)
        textViewRow.text = label

        textViewRow.setOnClickListener {
            val predictionsChipGroup = listDialog.parentActivity.predictionsChipGroup
            val classIdsAssigned = IntRange(0, predictionsChipGroup.childCount - 1)
                    .map { idx -> predictionsChipGroup[idx].id.toLong() }
            if (!classIdsAssigned.contains(classId)) {
                val chip = Chip(predictionsChipGroup.context)
                chip.id = classId.toInt()
                chip.text = label
                chip.isCloseIconVisible = true

                chip.setOnCloseIconClickListener {
                    predictionsChipGroup.removeView(it)
                    if (predictionsChipGroup.size > 0) {
                        listDialog.parentActivity.enableConfirmButton()
                    } else {
                        listDialog.parentActivity.disableConfirmButton()
                    }
                }
                predictionsChipGroup.addView(chip)
                listDialog.parentActivity.enableConfirmButton()
            }

            listDialog.parentActivity.getSystemService(Context.INPUT_METHOD_SERVICE).also { imm ->
                imm as InputMethodManager
                imm.hideSoftInputFromWindow(
                        (listDialog.currentFocus ?: View(listDialog.parentActivity)).windowToken, 0
                )
            }

            listDialog.dismiss()
        }

        return cv
    }

    fun hideIfNoMatch(s: String) {
        itemIndices.clear()
        var i = 0
        for (itemIdx in listItems.indices) {
            val label = listItems[itemIdx].second
            if (s.isEmpty() || label.contains(s, ignoreCase = true) || s.contains(label, ignoreCase = true)) {
                itemIndices[i] = itemIdx
                i += 1
            }
        }
    }
}
