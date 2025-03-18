package ai.onnxruntime.example.objectdetection

object BraillePostProcessor {

    private var capitalizeNext = false
    private var numberMode = false
    private var prefixDetected = false
    private var prefixText = ""

    // Reset state variables
    fun resetStates() {
        capitalizeNext = false
        numberMode = false
        prefixDetected = false
        prefixText = ""
    }

    // Apply capitalization if needed
    private fun applyCapitalization(text: String): String {
        return if (capitalizeNext) {
            capitalizeNext = false
            text.replaceFirstChar { it.uppercase() }
        } else {
            text
        }
    }

    // Convert letters to digits if in number mode
    private fun convertToDigit(text: String): String {
        return when (text) {
            "a" -> "1"
            "b" -> "2"
            "c" -> "3"
            "d" -> "4"
            "e" -> "5"
            "f" -> "6"
            "g" -> "7"
            "h" -> "8"
            "i" -> "9"
            "j" -> "0"
            else -> text
        }
    }

    // Check if a word is a whole word in Grade 2
    private fun isWholeWord(text: String): Boolean {
        val wholeWords = setOf(
            // One-cell whole words (alphabet and non-alphabet)
            "bakit", "kaniya", "dahil", "paano", "ganoon", "hindi", "ikaw", "hakbang", "kaya",
            "lamang", "mga", "ngayon", "para", "kailan", "rin", "sang-ayon", "tayo", "upang",
            "bagaman", "wala", "ito", "yaman", "sa", "ako", "anak", "ang", "araw", "at",
            "ay", "hanggang", "raw", "tunay", "kanila", "maging", "mahal", "na", "naging",
            "ng", "ibig", "ingay",

            // Two-cell whole words and non-alphabet
            "binata", "karaniwan", "dalaga", "ewan", "papaano", "gunita", "hapon", "isip",
            "halaman", "kailangan", "larawan", "mabuti", "noon", "opo", "patuloy", "kislap",
            "roon", "subalit", "talaga", "ugali", "buhay", "wasto", "eksamen", "ayaw", "salita",

            // Two-cell non-alphabet contractions
            "alam", "anggi", "bulaklak", "kabila", "masama", "nawa", "ngunit", "panahon",
            "sabi", "sinta", "tungkol", "ukol", "wakas"
        )
        return text in wholeWords
    }


    // Check if the word is a part-word contraction (Grade 2)
    private fun isPartWord(text: String): Boolean {
        val partWords = setOf(
            "an", "ang", "ar", "at", "aw", "er", "han", "ibig", "ing", "mag",
            "mahal", "nag", "ng", "pag", "tu"
        )
        return text in partWords
    }

    // Improved word accumulation logic
    fun formText(detectionResults: List<String>, grade: Int): String {
        resetStates()
        val finalText = StringBuilder()
        val currentWord = StringBuilder()

        for (text in detectionResults) {
            // Handle special characters (capital, number, prefix)
            if (text == "capital") {
                capitalizeNext = true
                continue
            }
            if (text == "number") {
                numberMode = true
                continue
            }

            // Apply capitalization or number mode
            val processedText = if (numberMode) convertToDigit(text) else applyCapitalization(text)

            if (grade == 1) {
                // Grade 1: Accumulate characters to form words
                currentWord.append(processedText)
            } else if (grade == 2) {
                // Grade 2: Handle whole words, part words, and contractions
                if (isWholeWord(processedText)) {
                    // Add a space before whole words if needed
                    if (currentWord.isNotEmpty()) {
                        finalText.append(currentWord).append(" ")
                        currentWord.clear()
                    }
                    finalText.append(processedText).append(" ")
                } else if (isPartWord(processedText)) {
                    // Combine part word with the previous or next word without space
                    currentWord.append(processedText)
                } else {
                    // Add regular characters or contractions
                    currentWord.append(processedText)
                }
            }
        }

        // Append any remaining word to the final text
        if (currentWord.isNotEmpty()) {
            finalText.append(currentWord)
        }

        return finalText.toString().trim()
    }

}
