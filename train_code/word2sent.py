import spacy

# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")

def convert_labels_to_sentence_level(response, labels):
    # Split the response into words based on spaces
    words = response.split(" ")

    # Process the response with SpaCy to split into sentences
    doc = nlp(response)
    sentences = list(doc.sents)

    # Track character positions of each word in the original text
    word_start_positions = []
    char_pos = 0
    for word in words:
        word_start_positions.append(char_pos)  # Starting char position of the word
        char_pos += len(word) + 1  # Move to next word position (+1 for the space)

    # Convert the word positions (from labels) to character-based positions
    label_positions = []
    for label in labels:
        start_word_pos = label['start']
        end_word_pos = label['end']

        # Ensure start and end positions are within valid ranges
        if start_word_pos >= len(words):
            print(f"Warning: Start position {start_word_pos} is out of range. Skipping.")
            continue
        if end_word_pos > len(words):
            end_word_pos = len(words)  # Clip to the maximum word length

        # Ensure end_word_pos is valid and not before start_word_pos
        if end_word_pos <= start_word_pos:
            print(f"Warning: End position {end_word_pos} is not valid. Skipping.")
            continue

        # Find the corresponding character positions
        start_char_pos = word_start_positions[start_word_pos]
        end_char_pos = word_start_positions[end_word_pos - 1] + len(words[end_word_pos - 1])

        label_positions.append({
            "label": label['label'],
            "start": start_char_pos,
            "end": end_char_pos
        })

    # Initialize a dictionary to store labeled sentences
    label_sentence_map = {label['label']: [] for label in label_positions}

    # Check for label overlaps with sentences based on character positions
    for sentence in sentences:
        sentence_start = sentence.start_char
        sentence_end = sentence.end_char
        sentence_text = sentence.text.strip()
        sentence_length = len(sentence_text)

        label_intersections = {label['label']: 0 for label in label_positions}

        for label in label_positions:
            # Calculate the overlap between the label and the sentence
            overlap_start = max(label['start'], sentence_start)
            overlap_end = min(label['end'], sentence_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                label_intersections[label['label']] += overlap

        # Assign the sentence to all labels that have significant overlap
        for label, overlap in label_intersections.items():
            if overlap / sentence_length > 0.65:  # Check if more than half of the sentence is covered
                label_sentence_map[label].append(sentence_text)

    return label_sentence_map

# Example usage
# response = "I think that uranium should not be mined at Mount Taylor.I think this because it could damage the environment really badly.There are multiple peices of evidence that support this claim.Some are "For every two pounds of uranium, 998 pounds of radioactive waste is left over in piles and pits"Over time it can leak into the soil and underground water if the area is not reclaimed.","Exposure to radioactive dust and contaminated groundwater causes health problems. Indeed, there is a higher incidence of lung cancer and other serious illnesses in mining communities. ".But there are also some reasons that it could help Mount Taylor. One reason is that it could pollute the soil.In paragraph 4 it says"Over time it can leak into the soil and underground water if the area is not reclaimed". This means that the soil can get damaged if uranium gets mined.This would be very bad for the plants.This is only one reason uranium should not get mined. Another reason that Uranium should not get mined is that the waste could effect the environment.The text says "For every two pounds of uranium, 998 pounds of radioactive waste is left over in piles and pits."This means that when uranium get mined loads of radioactive waste gets sent into the air.This could be very bad for the people breathing it in and everything living close by.This shows that it wouldn\'t be safe to mine uranium. The last but not least reason is, that being near the radioactive waste can cause serious health issues. In the text it says"Exposure to radioactive dust and contaminated groundwater causes health problems. Indeed, there is a higher incidence of lung cancer and other serious illnesses in mining communities. " This means that if miners are near the radioactive smells that they could get very sick.This could cause less people to want to work as a miner to stay safe.This shows that mining at Mount taylor could make many people sick. As you can probaly tell I would rather have uranium not mined at Mount taylor but there are a couple reasons why someone would want to.One is that it could bring many jobs to people. In paragraph 7 it says"The nearby city of Grants welcomes the opportunity to provide jobs and create business projects." This means that there could be many job opening for people if they need it. I think that this is very helpful, some people need a job and this is giving it to them. This proves that there are reasons that someone could want uranium mined at Mount taylor. In this essay I have gone over the reasons and evidence of why I think uranium shouldn\'t be mined at Mount Taylor.But it is also important to remember that there is always another side and it can have its reasons too. The reasons that show that uranium should\'t be mined at mount taylor are: that it could pollute the soil, is that the waste could effect the environment, and that being near the radioactive waste can cause serious health issues. But a reason that shows that uranium should be mined at Mount taylor is:it could bring many jobs to people. As I have said before I think that uranium shouldn\'t be mined at Mount taylor and I hope that the article has changed your mind if you originally thought otherwise."
