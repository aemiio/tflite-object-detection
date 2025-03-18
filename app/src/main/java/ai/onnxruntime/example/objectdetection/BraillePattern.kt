package ai.onnxruntime.example.objectdetection

//Dot 1: 100000
//Dot 2: 010000
//Dot 3: 001000
//Dot 4: 000100
//Dot 5: 000010
//Dot 6: 000001

object BraillePatternMatcher {

//    val grade1Dictionary = mapOf(
//        // Alphabet
//        "100000" to "a",
//        "110000" to "b",
//        "100100" to "c",
//        "100110" to "d",
//        "100010" to "e",
//        "110100" to "f",
//        "110110" to "g",
//        "110010" to "h",
//        "010100" to "i",
//        "010110" to "j",
//        "101000" to "k",
//        "111000" to "l",
//        "101100" to "m",
//        "101110" to "n",
//        "101010" to "o",
//        "111100" to "p",
//        "111110" to "q",
//        "111010" to "r",
//        "011100" to "s",
//        "011110" to "t",
//        "101001" to "u",
//        "111001" to "v",
//        "010111" to "w",
//        "101101" to "x",
//        "101111" to "y",
//        "101011" to "z",
//
//        // Composition signs
//        "000001" to "capital",
//        "001111" to "number"
//    )
//
//    // Grade 1 Two-Cell Dictionary
//    val grade1TwoCellDictionary = mapOf(
//        "000100-101110" to "ñ"  // Dot 4 + Dot 1345 (ñ)
//    )
//
//    // Grade 2 One-Cell Dictionary
//    val grade2OneCellDictionary = mapOf(
//        // Alphabet Contractions
//        "110000" to "bakit",
//        "100100" to "kaniya",
//        "100110" to "dahil",
//        "110100" to "paano",
//        "110110" to "ganoon",
//        "110010" to "hindi",
//        "010100" to "ikaw",
//        "010110" to "hakbang",
//        "101000" to "kaya",
//        "111000" to "lamang",
//        "101100" to "mga",
//        "101110" to "ngayon",
//        "111100" to "para",
//        "111110" to "kailan",
//        "111010" to "rin",
//        "011100" to "sang-ayon",
//        "011110" to "tayo",
//        "101001" to "upang",
//        "111001" to "bagaman",
//        "010111" to "wala",
//        "101101" to "ito",
//        "101111" to "yaman",
//        "101011" to "sa",
//
//        // Non-Alphabet Contractions
//        "010101" to "ako",
//        "100011" to "anak",
//        "011101" to "ang",
//        "001110" to "araw",
//        "001111" to "at",
//        "111101" to "ay",
//        "111011" to "hanggang",
//        "110111" to "raw",
//        "110011" to "tunay",
//        "100001" to "kanila",
//        "100101" to "maging",
//        "111111" to "mahal",
//        "011111" to "na",
//        "110101" to "naging",
//        "110001" to "ng",
//        "001100" to "ibig",
//        "001101" to "ingay"
//    )
//
//    // Grade 2 Part-Word Dictionary
//    val grade2PartWordDictionary = mapOf(
//        "100011" to "an",
//        "011101" to "ang",
//        "001110" to "ar",
//        "001111" to "at",
//        "010101" to "aw",
//        "110111" to "er",
//        "111011" to "han",
//        "001100" to "ibig",
//        "001101" to "ing",
//        "100101" to "mag",
//        "111111" to "mahal",
//        "110101" to "nag",
//        "110001" to "ng",
//        "100111" to "pag",
//        "110011" to "tu"
//    )
//
//    // Grade 2 Two-Cell Dictionary
//    val grade2TwoCellDictionary = mapOf(
//        // Alphabet Two-Cell
//        "000010-110000" to "binata",
//        "000010-100100" to "karaniwan",
//        "000010-100110" to "dalaga",
//        "000010-100010" to "ewan",
//        "000010-110100" to "papaano",
//        "000010-110110" to "gunita",
//        "000010-110010" to "hapon",
//        "000010-010100" to "isip",
//        "000010-010110" to "halaman",
//        "000010-101000" to "kailangan",
//        "000010-111000" to "larawan",
//        "000010-101100" to "mabuti",
//        "000010-101110" to "noon",
//        "000010-101010" to "opo",
//        "000010-111100" to "patuloy",
//        "000010-111110" to "kislap",
//        "000010-111010" to "roon",
//        "000010-011100" to "subalit",
//        "000010-011110" to "talaga",
//        "000010-101001" to "ugali",
//        "000010-111001" to "buhay",
//        "000010-010111" to "wasto",
//        "000010-101101" to "eksamen",
//        "000010-101111" to "ayaw",
//        "000010-101011" to "salita",
//
//        // Non-Alphabet Two-Cell
//        "000010-111101" to "alam",
//        "000010-011101" to "anggi",
//        "000010-001111" to "bulaklak",
//        "000010-100001" to "kabila",
//        "000010-100101" to "masama",
//        "000010-110101" to "nawa",
//        "000010-110001" to "ngunit",
//        "000010-100111" to "panahon",
//        "000010-100011" to "sabi",
//        "000010-001100" to "sinta",
//        "000010-110011" to "tungkol",
//        "000010-001101" to "ukol",
//        "000010-010101" to "wakas"
//    )

    fun getBrailleEntry(classId: Int, grade: Int): BrailleEntry? {
        val map = BrailleMap.getBrailleMap(grade)
        return map[classId]
    }

    // Get Meaning from Class ID and Grade
    fun getMeaning(classId: Int, grade: Int): String {
        return getBrailleEntry(classId, grade)?.meaning ?: "?"
    }

    // Get Binary from Class ID and Grade
    fun getBinary(classId: Int, grade: Int): String {
        return getBrailleEntry(classId, grade)?.binary ?: "?"
    }
}