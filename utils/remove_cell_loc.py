import re
import re
import json
import sys

def remove_cell_loc(doctag):
    # Only remove <loc_...> tags inside <otsl>...</otsl> blocks, leave others untouched
    def process_otsl_block(match):
        block = match.group(0)
        # Find all <loc_...> tags in the block
        loc_tags = list(re.finditer(r'<loc_\d+>', block))
        # If 4 or fewer, keep all
        if len(loc_tags) <= 4:
            return block
        # Otherwise, keep first 4, remove the rest
        result = ''
        last = 0
        for i, m in enumerate(loc_tags):
            if i < 4:
                # Keep this tag
                result += block[last:m.end()]
                last = m.end()
            else:
                # Skip this tag
                result += block[last:m.start()]
                last = m.end()
        result += block[last:]
        return result

    # Process <otsl>...</otsl> blocks only
    doctag = re.sub(r'<otsl>[\s\S]*?</otsl>', process_otsl_block, doctag)
    return doctag

def process_jsonl(input_path, output_path=None):
    with open(input_path, 'r', encoding='utf-8') as infile:
        out = open(output_path, 'w', encoding='utf-8') if output_path else None
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if 'doctag' in obj:
                    obj['doctag'] = remove_cell_loc(obj['doctag'])
                result = json.dumps(obj, ensure_ascii=False)
                if out:
                    out.write(result + '\n')
                else:
                    print(result)
            except Exception as e:
                print(f"Error processing line: {e}", file=sys.stderr)
        if out:
            out.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove <loc_...> tags from doctag in JSONL file.")
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', help='Output JSONL file (optional, prints to stdout if not set)')
    args = parser.parse_args()
    process_jsonl(args.input, args.output)