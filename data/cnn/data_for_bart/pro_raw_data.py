from collections import deque
import json
import os
from pydoc import doc
path = '../source/cnn/stories'
assert os.path.isdir(path)
documents = []
story_filenames_list = sorted(os.listdir(path))
for story_filename in story_filenames_list:
    if "summary" in story_filename:
        continue
    path_to_story = os.path.join(path, story_filename)
    if not os.path.isfile(path_to_story):
        continue
    documents.append(path_to_story)
print(len(documents))
# with open(documents[0]) as fin:
#     for line in fin:
#         print(line)


def process_story(raw_story):
    nonempty_lines = list(filter(lambda x: len(x) != 0, [line.strip() for line in raw_story.split("\n")]))

    # for some unknown reason some lines miss a period, add it
    nonempty_lines = [_add_missing_period(line) for line in nonempty_lines]

    # gather article lines
    story_lines = []
    lines = deque(nonempty_lines)
    while True:
        try:
            element = lines.popleft()
            if element.startswith("@highlight"):
                break
            story_lines.append(element)
        except IndexError:
            # if "@highlight" is absent from the file we pop
            # all elements until there is None, raising an exception.
            return story_lines, []

    # gather summary lines
    summary_lines = list(filter(lambda t: not t.startswith("@highlight"), lines))

    return story_lines, summary_lines


def _add_missing_period(line):
    END_TOKENS = [".", "!", "?", "...", "'", "`", '"', "\u2019", "\u2019", ")"]
    if line.startswith("@highlight"):
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + "."



with open("./stories.txt", "w") as fout:
    for k, document_path in enumerate(documents):
        document_name = document_path.split("/")[-1]
        with open(document_path, encoding="utf-8") as source:
            raw_story = source.read()
            story_lines, summary_lines = process_story(raw_story)
            fout.write(json.dumps({"id": k, "summary": summary_lines, "story": story_lines})+"\n")
