"""
new key_proc.py
lightly adapted version of this script:
https://github.com/wgantt/mucd/blob/main/scripts/preprocessing/proc_keys.py

"""
import os
import re
import json
import argparse
from codecs import decode
from collections import defaultdict


def cleankey(keystr):
    """
    Cleans the key string by removing non-alphabetic characters and converting to lowercase.
    """
    return re.sub(r"[^A-Z]+", "_", keystr).strip("_").lower()

def clean_docid(value):
    """
    Removes the parenthetical information from the docid.
    """
    return re.sub(r"\s*\(.*$", "", value)

ALL_KEYS = """
MESSAGE: ID
MESSAGE: TEMPLATE
INCIDENT: DATE
INCIDENT: LOCATION
INCIDENT: TYPE
INCIDENT: STAGE OF EXECUTION
INCIDENT: INSTRUMENT ID
INCIDENT: INSTRUMENT TYPE
PERP: INCIDENT CATEGORY
PERP: INDIVIDUAL ID
PERP: ORGANIZATION ID
PERP: ORGANIZATION CONFIDENCE
PHYS TGT: ID
PHYS TGT: TYPE
PHYS TGT: NUMBER
PHYS TGT: FOREIGN NATION
PHYS TGT: EFFECT OF INCIDENT
PHYS TGT: TOTAL NUMBER
HUM TGT: NAME
HUM TGT: DESCRIPTION
HUM TGT: TYPE
HUM TGT: NUMBER
HUM TGT: FOREIGN NATION
HUM TGT: EFFECT OF INCIDENT
HUM TGT: TOTAL NUMBER
""".strip().split(
    "\n"
)


LOCATION_RE = r"([\w ]+)(\(\w+ ?\w*\))*" 

ALL_KEYS = set(cleankey(k) for k in ALL_KEYS)

NON_LIST_VALUED_KEYS = """
message_id
message_template
message_template_optional
incident_type
incident_stage_of_execution
perp_incident_category
""".split()

SELECTED_KEYS = """
perp_individual_id
perp_organization_id
perp_organization_confidence
perp_incident_category
hum_tgt_name
hum_tgt_description
hum_tgt_effect_of_incident
hum_tgt_foreign_nation
hum_tgt_number
hum_tgt_total_number
hum_tgt_type
phys_tgt_id
phys_tgt_effect_of_incident
phys_tgt_foreign_nation
phys_tgt_number
phys_tgt_total_number
phys_tgt_type
incident_instrument_id
incident_instrument_type
incident_location
incident_date
incident_stage_of_execution
""".split() 

MODIFIED_KEYS = """
perp_individual_id
perp_organization_id
perp_organization_confidence
perp_incident_category
hum_tgt_name
hum_tgt_description
hum_tgt_effect_of_incident
hum_tgt_number
hum_tgt_type
phys_tgt_id
phys_tgt_effect_of_incident
phys_tgt_number
phys_tgt_type
incident_instrument_id
incident_instrument_type
incident_location
incident_date
incident_stage_of_execution
""".split()

EXTRACTIVE_FIELDS = """
incident_instrument_id
perp_individual_id
perp_organization_id
phys_tgt_id
hum_tgt_name
""".split()

INCIDENT_TYPES = [
"ATTACK",
"ARSON",
"KIDNAPPING",
"ROBBERY",
"FORCED WORK STOPPAGE",
"BOMBING",
"ATTACK / BOMBING",
"BOMBING / ATTACK",
"*"
]       

SET_FILL_KEYS_ALLOWED_VALUES = {
    "incident_stage_of_execution": {"ACCOMPLISHED", "ATTEMPTED", "THREATENED"},
    "perp_incident_category": {
        "TERRORIST ACT",
        "STATE-SPONSORED VIOLENCE",
        "? TERRORIST ACT",
        "? STATE-SPONSORED VIOLENCE",
    },
    "perp_organization_confidence": {
        "REPORTED AS FACT",
        "ACQUITTED",
        "CLAIMED OR ADMITTED",
        "SUSPECTED OR ACCUSED",
        "SUSPECTED OR ACCUSED BY AUTHORITIES",
        "POSSIBLE",
    },
    "hum_tgt_effect_of_incident": {
        "DEATH",
        "NO DEATH",
        "INJURY",
        "NO INJURY",
        "NO INJURY OR DEATH",
        "REGAINED FREEDOM",
        "ESCAPED",
        "RESIGNATION",
        "NO RESIGNATION",
        "PROPERTY TAKEN FROM TARGET",
    },
    "phys_tgt_effect_of_incident": {
        "DESTROYED",
        "SOME DAMAGE",
        "NO DAMAGE",
        "MONEY TAKEN FROM TARGET",
        "PROPERTY TAKEN FROM TARGET",
        "TARGET TAKEN",
        None,  # Not sure why this seems to be the only set-fill value that needs a none slot...
    },
}

SELECTED_KEYS = set(SELECTED_KEYS)

assert SELECTED_KEYS <= ALL_KEYS

cur_docid = None

def warning(s):
    """
    Prints a warning message with the current document ID.
    """
    global cur_docid
    print(f"WARNING docid={cur_docid} | {s}")


def yield_keyvals(chunk):
    """
    Processes the raw MUC "key file" format.  Parses one entry ("chunk").
    Yields a sequence of (key,value) pairs.
    A single key can be repeated many times.
    This function cleans up key names, but passes the values through as-is.
    """
    curkey = None
    for line in chunk.split("\n"):
        if line.startswith(";"):
            yield "comment", line
            continue
        middle = 33  ## Different in dev vs test files... this is the minimum size to get all keys.
        keytext = line[:middle].strip()
        valtext = line[middle:].strip()
        if not keytext:
            ## it's a continuation
            assert curkey
        else:
            curkey = cleankey(keytext)
            assert curkey in ALL_KEYS

        yield curkey, valtext


        
def parse_country_location(location_expr: str):
    
    """
    Parses the location expression and returns a list of dictionaries representing the location.
    """
    error_list = ["CARAHUAICHI", "MORENO", "CITY", "CARAHUAICHI", "URABA", "PERQUIN", "USULUTAN", "JURISDICTION", "DEPARTMENT", "COUNTY", "MUNICIPALITY", "LIMA", "SAN SALVADOR", "WASHINGTON D"]
    out = []
    seen = set()
    for loc1 in location_expr.split(" / "):
        loc1 = loc1.strip()
        for loc2 in loc1.split(":"):
            loc2 = loc2.strip()
            if loc2[0] == "(" and loc2[-1] == ")":
                loc2 = loc2[1:-1]
            for loc3 in loc2.split("-"):
                loc3 = loc3.strip()
                if loc3.startswith("? "):
                    loc3 = loc3[2:]
                match = re.search(LOCATION_RE, loc3)
                assert match is not None
                groups = match.groups()
                assert len(groups) == 2
                g0 = groups[0].strip()
                if groups[1] is None:
                    if g0 not in seen and g0 not in error_list:
                        out.append({"strings": [g0]})
                        seen.add(g0)
                else:
                    assert groups[1][0] == "(" and groups[1][-1] == ")"
                    g1 = groups[1][1:-1]
                    if g0 + g1 not in seen:
                      
                        seen.add(g0 + g1)
                        seen.add(g0)
                        seen.add(g1)
    return out



def parse_city_location(location_expr: str):
    """
    Parses the location expression and returns a list of dictionaries representing the location.
    """
    out = []
    seen = set()
    for loc1 in location_expr.split(" / "):
        loc1 = loc1.strip()
        for loc2 in loc1.split(":"):
            loc2 = loc2.strip()
            if loc2[0] == "(" and loc2[-1] == ")":
                loc2 = loc2[1:-1]
            for loc3 in loc2.split("-"):
                loc3 = loc3.strip()
                if loc3.startswith("? "):
                    loc3 = loc3[2:]
                match = re.search(LOCATION_RE, loc3)
                assert match is not None
                groups = match.groups()
                assert len(groups) == 2
                g0 = groups[0].strip()
                if groups[1] is None:
                    if g0 not in seen:
                        #out.append({"type": "simple_strings", "strings": [g0]})
                        seen.add(g0)
                else:
                    assert groups[1][0] == "(" and groups[1][-1] == ")"
                    g1 = groups[1][1:-1]
                    if g0 not in seen:
                        if g1 in {'CITY', 'TOWN'}:
                        
                            out.append({"strings": [g0]})
                            seen.add(g0)
    if len(out) > 0:
        return out
    return None
       

def parse_location(location_expr: str):
    """
    Parses the location expression and returns a list of dictionaries representing the location.
    """
    exclude_country = ["CARAHUAICHI", "MORENO", "CITY", "CARAHUAICHI", "URABA", "PERQUIN", "USULUTAN", "JURISDICTION", "DEPARTMENT", "COUNTY", "MUNICIPALITY", "LIMA", "SAN SALVADOR", "WASHINGTON D"]
    out = []
    seen = set()
    for loc1 in location_expr.split(" / "):
        loc1 = loc1.strip()
        for loc2 in loc1.split(":"):
            loc2 = loc2.strip()
            if loc2[0] == "(" and loc2[-1] == ")":
                loc2 = loc2[1:-1]
            for loc3 in loc2.split("-"):
                loc3 = loc3.strip()
                if loc3.startswith("? "):
                    loc3 = loc3[2:]
                match = re.search(LOCATION_RE, loc3)
                assert match is not None
                groups = match.groups()
                assert len(groups) == 2
                g0 = groups[0].strip()
                if groups[1] is None:
                    if g0 not in seen and g0 not in exclude_country:
                        out.append({"strings": [g0]})
                        seen.add(g0)
                else:
                    assert groups[1][0] == "(" and groups[1][-1] == ")"
                    g1 = groups[1][1:-1]
                    if g0 + g1 not in seen:
                        out.append(
                            {
                                "type": "colon_clause",
                                "strings_lhs": [g0],
                                "strings_rhs": [g1],
                            }
                        )
                        seen.add(g0 + g1)
    return out


def parse_values_modified(keyvals, selected_keys=None, incident_types=None, incident_location_type=None):
    """
    Takes key,value pairs as input, where the values are unparsed.
    Checks if the incident_type is in the incident_types set:
    if not:
         "message_id": "TST4-MUC4-0002",
        "message_template": "*",
         "incident_type": "*"
    else:
        Regular parsing 
        Filter down to the slots we want, and parse their values.
    """
    if selected_keys is None:
        selected_keys = SELECTED_KEYS
        
    if incident_types is None:
        incident_types == set(INCIDENT_TYPES)
    for key, value in keyvals:
        if key == "incident_type":
            if value in incident_types:
                return parse_values(keyvals, selected_keys, incident_types, incident_location_type)
            else:
                return parse_values_exclude_event(keyvals, selected_keys, incident_types)
                break

def parse_values_exclude_event(keyvals, selected_keys=None, incident_types=None):
    """
    Retruns a non-event for excluded event types
    """
    if selected_keys is None:
        selected_keys = SELECTED_KEYS
        
    if incident_types is None:
        incident_types = INCIDENT_TYPES
       
    for key, value in keyvals: 
        if key == "message_id":
            yield key, clean_docid(value)
            continue
        if key == "message_template":
            if re.search(r"^\d+$", value):
                yield key, int(value)
            elif value == "*":
                yield key, value
            elif re.search(r"^\d+ \(OPTIONAL\)$", value):
                yield key, int(value.split()[0])
                yield "message_template_optional", True
            else:
                assert False, "bad message_template format"
            continue
        if key == "incident_type":
            yield key, '*'
            break
        else:
            continue
        
def parse_values(keyvals, selected_keys=None, incident_types=None, incident_location_type=None):
    """
    Takes key,value pairs as input, where the values are unparsed.
    Filter down to the fields we want, and parse their values as well.
    """
    if selected_keys is None:
        selected_keys = SELECTED_KEYS
        
    if incident_types is None:
        incident_types = INCIDENT_TYPES
    
   
    for key, value in keyvals:
        if key == "message_id":
            yield key, clean_docid(value)
            continue
     
        if key == "incident_type":
            if value in incident_types:
                yield key, value
            else:
                yield key, '*'
                break

        if key in selected_keys:
            if value == "*":
                continue

            if value == "-":
                yield key, None
                continue

            if key == "incident_location":
                if incident_location_type == "city":
                    
                    yield "incident_location_country", parse_country_location(value)
                    yield "incident_location_city", parse_city_location(value)
                else:
                    yield key, parse_location(value)
                continue

            if '"' not in value:
                if key not in {
                    "incident_date",
                    "incident_stage_of_execution",
                    "perp_incident_category",
                    "perp_organization_confidence",
                }:
                    warning(
                        f"apparent data error, missing quotes. adding back in. value was ||| {value}"
                    )
                    value = '"' + value + '"'

            value = parse_one_value(value, key)
            if key in SET_FILL_KEYS_ALLOWED_VALUES:
                strings_key = "strings" if "strings" in value else "strings_lhs"
                for v in value[strings_key]:
                    assert v in SET_FILL_KEYS_ALLOWED_VALUES[key], f"{key}: {v}"

            yield key, value


def parse_one_value(namestr, slotname=None):
    """
    Returns a dictionary with 
    "strings" ==> lists of strings for simple strings and "string_lhs". 
    "the reference string "string_rhs" is not included. 

    """

    global cur_docid
    # Fix bugs in the data
    if cur_docid == "DEV-MUC3-0604" and "BODYGUARD OF EL ESPECTADOR" in namestr:
        # DEV-MUC3-0604 (MDESC)
        # ? ("BODYGUARD OF EL ESPECTADOR'S CHIEF OF DISTRIBUTION IN MEDELLIN" / "BODYGUARD"): "PEDRO LUIS OSORIO"
        namestr = '''? "BODYGUARD OF EL ESPECTADOR'S CHIEF OF DISTRIBUTION IN MEDELLIN" / "BODYGUARD" / "PEDRO LUIS OSORIO"'''
    if namestr == 'MACHINEGUNS"':
        # DEV-MUC3-0217
        namestr = '"' + namestr

    d = {}
    match = re.search(r"\? *(.*)", namestr)
    if match:
        d["optional"] = True
        namestr = match.group(1)

    if ":" in namestr:
        assert len(re.findall(":", namestr)) == 1
        lhs, rhs = re.split(r" *: *", namestr)
        if lhs[0] == "(":
            lhs = lhs[1:]
        if lhs[-1] == ")":
            lhs = lhs[:-1]
        #rhs_value = parse_strings_possibly_with_alternations(rhs)
        lhs_value = parse_strings_possibly_with_alternations(lhs, slotname)
        d.update(
            {"strings": lhs_value}
        )
        return d

    else:
        strings = parse_strings_possibly_with_alternations(namestr, slotname)
        d.update({"strings": strings})
        return d


def parse_strings_possibly_with_alternations(namestr, slotname=None):
    namestr = namestr.strip()
    assert ":" not in namestr, namestr
    assert not namestr.startswith("?")
    parts = re.split(" */ *", namestr)
    parts = [ss.strip() for ss in parts]
    strings = []
    for ss in parts:
        if ss == "-":
            # We should see this only inside a colon clause. There are a few of these, e.g.
            # 21. HUM TGT: NUMBER                 -: "ORLANDO LETELIER"
            strings.append(None)
            continue
        if slotname in {
            "hum_tgt_effect_of_incident",
            "phys_tgt_effect_of_incident",
            "incident_stage_of_execution",
            "perp_incident_category",
            "perp_organization_confidence",
        }:
            # These slots should not have strings escaped
            assert ss in SET_FILL_KEYS_ALLOWED_VALUES[slotname], f"{slotname}: {ss}"
        
        elif slotname == "incident_date":
            if ss[0] == "(":
                ss = ss[1:]
            if ss[-1] == ")":
                ss = ss[:-1]
        else:
            if (ss[0] == '"' and ss[-1] != '"') or (ss[0] != '"' and ss[-1] == '"'):
                warning("WTF ||| " + ss)
            if ss[0] == '"':
                ss = ss[1:]
            if ss[-1] == '"':
                ss = ss[:-1]
        # ss = ss.decode('string_escape')  # They seem to use C-style backslash escaping
        ss = decode(ss, "unicode-escape")
        ss = ss.strip()
        strings.append(ss)
    return strings


def test_parsestrings():
    """
    Tests the parse_strings_possibly_with_alternations function.
    """
    f = parse_strings_possibly_with_alternations
    s = '"CAR DEALERSHIP"'
    assert set(f(s)) == {"CAR DEALERSHIP"}
    s = '"TUPAC AMARU REVOLUTIONARY MOVEMENT" / "MRTA"'
    assert set(f(s)) == {"TUPAC AMARU REVOLUTIONARY MOVEMENT", "MRTA"}


def test_parse_one_value():
    """
    Tests the parse_one_value function.
    """
    s = '"U.S. JOURNALIST": "BERNARDETTE PARDO"'
    d = parse_one_value(s)
    assert d["strings"] == ["U.S. JOURNALIST"]


def fancy_json_print(keyvals):
    """
    Prints the keyvals in a fancy JSON format.
    """
    lines = [json.dumps(kv, sort_keys=True) for kv in keyvals]
    s = ""
    s += "[\n  "
    s += ",\n  ".join(lines)
    s += "\n]"
    return s


def keyvals_to_dict(keyvals):
    """
    Converts the keyvals to a dictionary.
    """
    out = defaultdict(list)
    for (k, v) in keyvals:
        if v is None or k == "incident_location" or k == "incident_location_city" or k == "incident_location_country" or k in NON_LIST_VALUED_KEYS:
            out[k] = v
        else:
            assert isinstance(v, dict), f"{k}: {v}"
            if out[k] is None:
                out[k] = []
            out[k].append(v)
    return out



if __name__ == "__main__":
    """
    Main function to process the MUC keyfiles and output the processed data in JSON format.
    """

    p = argparse.ArgumentParser()
    p.add_argument("--input", help="the raw MUC keyfiles to be processed")
    p.add_argument("--output", help="the JSON file where the output will be written")
    p.add_argument("--dataset_type", type=str, default=False, help="extractive(five fields), modified(remove some fields)")
    p.add_argument("--event_type", type=str, default=None, help="the event types to be included")
    p.add_argument("--incident_location_type", type=str, default=None, help="the incident_location_type to be included")
    args = p.parse_args()
    
    if args.event_type:
        args.event_type = {t.strip() for t in args.event_type.split(",")}
        incident_types = set(args.event_type)
        print('incident_types', incident_types)
    else: 
        incident_types = set(INCIDENT_TYPES)
        print('incident_types', incident_types)

    if args.dataset_type == "extractive":
        selected_keys = EXTRACTIVE_FIELDS
        print('selected_keys', selected_keys)
        
    elif args.dataset_type == "modified":
        selected_keys = MODIFIED_KEYS
        print('selected_keys', selected_keys)
    else:
        selected_keys = SELECTED_KEYS
        print('selected_keys', selected_keys)

    if os.path.isfile(args.input):
        keyfiles = [args.input]
    elif os.path.isdir(args.input):
        path = os.path.abspath(args.input)
        keyfiles = [
            os.path.join(path, f)
            for f in os.listdir(args.input)
            if f.startswith("key-")
        ]
    else:
        raise ValueError("Could not find input file or directory")
    assert keyfiles, f"No keyfiles found"
    
    lines = []
    for keyfile in keyfiles:
        with open(keyfile) as f:
            for line in f:
                l = line.rstrip()
                if not re.search(r"^\s*;", l):
                    lines.append(l)
    data = "\n".join(lines)
    chunks = re.split(r"\n\n+|\n(?=0\. )", data) 
    chunks = [c.strip() for c in chunks if c.strip()]

    output = defaultdict(list)
    
    # each chunck is a template
    for chunk in chunks:
        keyvals1 = list(yield_keyvals(chunk))
        #print('keyvals1', keyvals1)
        #print('\n\n\n')
        assert all(k in ALL_KEYS or k == "comment" for k, v in keyvals1)
        cur_docid = clean_docid(dict(keyvals1)["message_id"]) 
        #we need to add a check to see if the incident_type in in the incident_type set, 
        # if not, different parse_function
        keyvals2 = list(parse_values_modified(keyvals1, selected_keys=selected_keys, incident_types=incident_types, incident_location_type=args.incident_location_type))
        #keyvals2 = list(parse_values(keyvals1, selected_keys=selected_keys, incident_types=incident_types))
        keyvals_dict = keyvals_to_dict(keyvals2)
        output[keyvals_dict["message_id"]].append(keyvals_dict)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)