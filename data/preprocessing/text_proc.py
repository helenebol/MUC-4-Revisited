"""
Adapted version of this script:
https://github.com/wgantt/mucd/blob/main/scripts/preprocessing/proc_texts.py
"""
import argparse
import json
import os
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to file or directory")
    parser.add_argument("--split", help="split to process")
    parser.add_argument("--output", help="path to output folder")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        texts = [args.input]
    elif os.path.isdir(args.input):
        path = os.path.abspath(args.input)
        texts = [os.path.join(path, f) for f in os.listdir(args.input)]
    else:
        raise ValueError("Can not find input file or directory")
    assert texts, f"No texts found in {args.input}"

    output = {}
    # loop over texts to extract docids, datelines, tags, and text
    for text in texts:
        doc_infos = []
        with open(text) as f:
            data = f.read()
            matches = list(re.finditer(r"(DEV-\S+) *\(([^\)]*)\)", data))
            has_source = bool(matches)
            if not matches:
                matches = list(re.finditer(r"(TST\d+-\S+)", data))

        for match in matches:
            docid = match.group(1)
            d = {
                "docid": docid,
                "char_start": match.end(),
                "char_before": match.start(),
            }
            if has_source:
                d["source"] = match.group(2)
            doc_infos.append(d)

        for i in range(len(doc_infos) - 1):
            doc_infos[i]["char_end"] = doc_infos[i + 1]["char_before"]
        doc_infos[-1]["char_end"] = len(data)

        for d in doc_infos:
            raw_text = data[d["char_start"] : d["char_end"]].strip()

            # issue: there are sometimes recursive (multiple?) datelines.  we only get the first in that case.

            tag_re = r"\[[^\]]+\]"
            tags_re = r"(?:%s\s+)+" % tag_re
            full_re = r"^(.*?)--\s+(%s)(.*)" % tags_re
            m = re.search(full_re, raw_text, re.DOTALL)
            if not m:
                print(f"Error processing document {d['docid']}")
                print("Could not find expected pattern in text:")
                print("First 200 characters of raw_text:")
                print("\nExpected format: <dateline> -- [TAG1] [TAG2] ... <text>")
                print(raw_text[:1000])
                #continue  # Skip this document instead of failing
                assert False  # Removing the hard assertion
                
               

            dateline = m.group(1).replace("\n", " ").strip()
            tags = m.group(2).replace("\n", " ")
            text = m.group(3)

            assert tags.upper() == tags
            tags = re.findall(tag_re, tags)
            tags = [x.lstrip("[").rstrip("]").lower() for x in tags]

            d["dateline"] = dateline
            d["tags"] = tags

            text = text.strip()
            text = text.replace("[", "(").replace("]", ")")

            d["text"] = text
            output[d["docid"]] = d

    #write the processed text to a file in the processed folder and create a folder in outputfolder with the split name
    # and add a folder within outputfolder called args.split 
    output_folder = os.path.abspath(args.output) 
    os.makedirs(output_folder, exist_ok=True)
    output_folder = os.path.join(output_folder, args.split)
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f"{args.split}.json")
    print(f"Writing output to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Successfully wrote {len(output)} documents") 
    
        