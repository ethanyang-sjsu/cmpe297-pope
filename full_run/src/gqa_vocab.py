"""GQA object vocabulary for POPE questions: DRAFT for team review (2026-11-15).

GQA scene graphs use ~1,500 free-form object names, so "this name is not in the scene
graph" is weak evidence that the object is absent. Each vocabulary object therefore has:

  SYNONYMS[obj]  names that mean the object itself. Used both to make "yes" questions
                 (the object is present if any synonym is in the scene graph) and to block
                 "no" questions. Kept precise: a wrong synonym creates a false "yes".
  BLOCKERS[obj]  extra names whose presence makes "no" unsafe: broader terms ("vehicle",
                 "animal", "food"), unambiguous visible parts ("tire", "beak", "hand"),
                 look-alikes and near-synonyms ("sofa"/"couch" are synonyms, but "mug" for
                 "cup" or "desk" for "table" are only blockers). Kept generous: an extra
                 blocker only shrinks the pool of absent objects to ask about.

Matching is on exact, lower-cased names; plural forms (+s, +es) of every listed name are
added automatically. Never substring-match: "bus" would hit "bush", "cat" "catcher",
"dog" "hot dog".

Left out on purpose:
- regions and materials: sky, wall, ground, grass, water, road, snow, floor, field
- object parts: head, hand, wheel, window, door, roof, leg, tail
- clothing and worn accessories: shirt, pants, hat, helmet, glasses, shoe, tie, watch
- ambiguous words: glass (window glass vs drinking glass), pot (flower vs cooking),
  cone (traffic vs ice cream), sign/pole/post (too generic), plural-only names such as
  skis and scissors (awkward in "Is there a ... ?")

Run `python src/gqa_vocab.py` for a coverage report over the GQA val scene graphs.
"""
import json
import os
import sys
from collections import Counter

# single-word names only in the .split() strings; multi-word names go in explicit lists
PERSON_PARTS_AND_CLOTHES = """
    hand finger thumb arm elbow shoulder wrist leg knee foot feet head face hair beard
    mustache ear nose mouth lips teeth tongue eye neck shirt t-shirt pants jeans shorts
    jacket coat sweater dress skirt suit vest uniform jersey wetsuit hat cap helmet
    glasses sunglasses goggles glove shoe sneakers boot sock scarf tie belt necklace
    bracelet earring wristband watch outfit clothes pitcher statue biker cowboy
    driver sandal headband mask""".split() + ["eye glasses", "cowboy hat", "bus driver", "tank top"]
ANIMALS = ["dog", "cat", "horse", "bird", "elephant", "giraffe", "zebra", "bear", "cow", "sheep"]
FUR_PARTS = ["paw", "fur", "whiskers", "collar", "leash", "snout", "muzzle"]
ROAD_VEHICLE_PARTS = ["license plate", "windshield", "windshield wiper", "bumper",
                      "headlight", "tail light", "taillight", "tire", "wheel", "hubcap"]

SYNONYMS = {
    # people
    "person": """person people man men woman women boy girl child children kid baby lady
        guy player skier surfer skater skateboarder snowboarder catcher batter umpire
        crowd spectator pedestrian tourist couple""".split()
        + ["baseball player", "soccer player", "tennis player"],
    # vehicles
    "car": ["car", "taxi", "suv", "jeep", "sedan", "police car"],
    "truck": ["truck", "fire truck", "food truck", "pickup truck"],
    "bus": ["bus", "school bus", "double decker bus"],
    "train": ["train", "locomotive", "train car"],
    "airplane": ["airplane", "plane", "jet", "aircraft", "airliner", "jet plane"],
    "boat": ["boat", "ship", "canoe", "kayak", "sailboat"],
    "motorcycle": ["motorcycle", "motorbike", "motor bike"],
    "bicycle": ["bicycle", "bike"],
    # animals
    "dog": ["dog", "puppy"],
    "cat": ["cat", "kitten"],
    "horse": ["horse", "pony", "foal"],
    "bird": ["bird", "duck", "pigeon", "goose", "geese", "swan", "parrot", "seagull", "gull"],
    "elephant": ["elephant"],
    "giraffe": ["giraffe"],
    "zebra": ["zebra"],
    "bear": ["bear", "polar bear", "panda bears", "panda"],
    "cow": ["cow", "cattle", "bull", "calf"],
    "sheep": ["sheep", "lamb"],
    # furniture
    "chair": ["chair", "folding chair", "office chair", "beach chair", "armchair"],
    "table": ["table", "coffee table", "end table", "side table", "picnic table", "dining table"],
    "bench": ["bench"],
    "bed": ["bed"],
    "couch": ["couch", "sofa"],
    "desk": ["desk", "computer desk"],
    "shelf": ["shelf", "shelves", "bookshelf", "bookshelves", "bookcase"],
    "cabinet": ["cabinet", "cupboard", "medicine cabinet"],
    # household
    "lamp": ["lamp", "floor lamp", "desk lamp", "table lamp"],
    "clock": ["clock", "alarm clock"],
    "mirror": ["mirror"],
    "pillow": ["pillow"],
    "blanket": ["blanket"],
    "curtain": ["curtain", "shower curtain"],
    "vase": ["vase"],
    "basket": ["basket", "waste basket"],
    "candle": ["candle"],
    "towel": ["towel", "bath towel", "kitchen towel"],
    "book": ["book"],
    "box": ["box", "tissue box", "pizza box", "cereal box", "lunch box", "storage box"],
    # kitchen and bathroom
    "bowl": ["bowl", "mixing bowl"],
    "plate": ["plate"],
    "cup": ["cup", "coffee cup"],
    "bottle": ["bottle", "water bottle", "wine bottle", "beer bottle", "soda bottle",
               "soap bottle", "ketchup bottle", "mustard bottle", "spray bottle", "shampoo bottle"],
    "fork": ["fork"],
    "knife": ["knife", "knives", "butter knife"],
    "spoon": ["spoon"],
    "napkin": ["napkin"],
    "tray": ["tray", "serving tray"],
    "sink": ["sink"],
    "faucet": ["faucet", "tap"],
    "refrigerator": ["refrigerator", "fridge"],
    "oven": ["oven"],
    "microwave": ["microwave", "microwave oven"],
    "stove": ["stove", "gas stove"],
    "toilet": ["toilet"],
    # electronics
    "laptop": ["laptop"],
    "keyboard": ["keyboard"],
    "computer mouse": ["computer mouse", "mouse"],
    "television": ["television", "tv"],
    "phone": ["phone", "cell phone", "cellphone", "telephone", "smartphone"],
    "remote control": ["remote control", "remote"],
    "camera": ["camera", "video camera"],
    # bags and sports
    "umbrella": ["umbrella", "beach umbrella", "parasol"],
    "backpack": ["backpack"],
    "bag": ["bag", "handbag", "purse", "shopping bag", "plastic bag", "trash bag"],
    "suitcase": ["suitcase"],
    "kite": ["kite"],
    "frisbee": ["frisbee"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "snowboard": ["snowboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "ball": ["ball", "tennis ball", "soccer ball", "baseball", "basketball", "football", "volleyball"],
    "baseball bat": ["baseball bat", "bat"],
    # street
    "traffic light": ["traffic light", "traffic signal", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "street light": ["street light", "streetlight", "lamp post", "lamppost", "light pole"],
    "stop sign": ["stop sign"],
    "flag": ["flag", "american flag"],
    "bridge": ["bridge"],
    "tent": ["tent"],
    "building": ["building", "apartment building", "skyscraper"],
    "house": ["house"],
    "tree": ["tree", "palm tree", "pine tree"],
    # food
    "pizza": ["pizza", "pizza slice", "pizza pie"],
    "cake": ["cake"],
    "banana": ["banana", "banana bunch"],
    "orange": ["orange"],
    "apple": ["apple"],
    "broccoli": ["broccoli"],
    "sandwich": ["sandwich"],
    "donut": ["donut", "doughnut"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "bread": ["bread", "bread loaf", "loaf"],
    # plants and misc
    "plant": ["plant", "potted plant", "houseplant"],
    "flower": ["flower"],
    "statue": ["statue", "sculpture"],
    "bucket": ["bucket", "pail"],
    "ladder": ["ladder"],
}

BLOCKERS = {
    "person": PERSON_PARTS_AND_CLOTHES + ["mannequin", "rider", "worker", "chef", "cook"],
    # vehicles
    "car": ["vehicle", "van", "toy car", "car door", "trunk"] + ROAD_VEHICLE_PARTS,
    "truck": ["vehicle", "van", "trailer"] + ROAD_VEHICLE_PARTS,
    "bus": ["vehicle", "bus stop", "van"] + ROAD_VEHICLE_PARTS,
    "train": ["vehicle", "caboose", "railroad car", "tram", "trolley", "subway"],
    "airplane": ["wing", "cockpit", "propeller", "jet engine", "engine", "landing gear", "tail fin", "tail"],
    "boat": ["sail", "mast", "dock", "yacht", "raft", "ferry"],
    "motorcycle": ["vehicle", "scooter", "moped", "tire", "wheel", "headlight", "license plate"],
    "bicycle": ["biker", "cyclist", "tire", "wheel", "handlebars", "handlebar", "pedal"],
    # animals: generic terms, toys, parts shared with other animals
    **{a: ["animal", "stuffed animal", "toy", "statue", "figurine", "tail"] for a in ANIMALS},
    # furniture
    "chair": ["seat", "stool", "furniture", "sofa", "couch"],
    "table": ["desk", "tablecloth", "furniture", "counter", "nightstand"],
    "bench": ["seat", "furniture"],
    "bed": ["headboard", "bedspread", "bedding", "mattress", "bedroom", "furniture"],
    "couch": ["cushion", "loveseat", "furniture"],
    "desk": ["table", "furniture"],
    "shelf": ["furniture", "rack"],
    "cabinet": ["cabinet door", "drawer", "furniture", "dresser"],
    # household
    "lamp": ["lamp shade", "light fixture", "light bulb"],
    "clock": ["clock tower", "clock hand", "clock face"],
    "mirror": ["side mirror", "rear view mirror"],
    "pillow": ["pillowcase", "cushion"],
    "blanket": ["comforter", "bedspread", "quilt", "bedding"],
    "curtain": ["drape", "blinds"],
    "vase": ["flower pot", "pot"],
    "basket": ["bin", "trash can"],
    "candle": ["candle holder"],
    "towel": ["paper towel", "towel dispenser", "napkin"],
    "book": ["bookshelf", "bookcase", "magazine", "notebook"],
    "box": ["container", "carton", "crate"],
    # kitchen and bathroom
    "bowl": ["dish", "toilet bowl", "container"],
    "plate": ["dish", "saucer", "platter"],
    "cup": ["mug", "coffee mug", "beer mug", "glass", "teacup", "cupcake"],
    "bottle": ["bottle cap", "jar", "container"],
    "fork": ["utensil", "silverware", "cooking utensils"],
    "knife": ["utensil", "silverware", "cooking utensils", "knife block"],
    "spoon": ["utensil", "silverware", "cooking utensils"],
    "napkin": ["paper towel", "tissue", "napkin dispenser", "towel"],
    "tray": ["platter", "baking sheet", "pan"],
    "sink": ["faucet", "counter"],
    "faucet": ["sink", "bathtub", "shower"],
    "refrigerator": ["freezer", "magnet", "magnets"],
    "oven": ["stove", "oven door", "microwave oven", "toaster oven", "pizza oven"],
    "microwave": ["oven", "toaster oven"],
    "stove": ["oven", "burner", "stovetop"],
    "toilet": ["toilet seat", "toilet bowl", "toilet tank", "toilet lid", "toilet brush"],
    # electronics
    "laptop": ["computer", "screen", "monitor", "computer monitor", "desktop computer"],
    "keyboard": ["laptop", "computer", "desktop computer"],
    "computer mouse": ["mouse pad", "computer"],
    "television": ["screen", "monitor", "computer monitor", "tv stand", "display"],
    "phone": ["screen", "device"],
    "remote control": ["controller", "game controller", "wii remote", "device"],
    "camera": ["device", "lens"],
    # bags and sports
    "umbrella": ["canopy"],
    "backpack": ["bag", "luggage"],
    "bag": ["backpack", "suitcase", "luggage", "sack", "pouch"],
    "suitcase": ["luggage", "bag", "luggage cart"],
    "kite": ["kite string"],
    "frisbee": ["disc"],
    "skateboard": ["board", "skateboarder"],
    "surfboard": ["board", "surfer"],
    "snowboard": ["board", "snowboarder"],
    "tennis racket": [],
    "ball": ["balloon"],
    "baseball bat": ["batter"],
    # street
    "traffic light": ["signal", "light"],
    "fire hydrant": [],
    "street light": ["lamp", "light", "pole"],
    "stop sign": ["sign", "street sign", "traffic sign", "sign post"],
    "flag": ["banner"],
    "bridge": ["overpass"],
    "tent": ["canopy"],
    "building": ["house", "tower", "clock tower", "store", "shop", "hotel", "station",
                 "train station", "church", "cathedral", "castle", "barn", "garage", "home"],
    "house": ["building", "home", "cabin", "barn", "hut", "cottage"],
    "tree": ["tree trunk", "tree branch", "branch", "trunk", "leaves", "leaf", "tree leaves",
             "foliage", "bush", "palm", "evergreen"],
    # food
    "pizza": ["pizza crust", "crust", "pepperoni", "cheese", "sauce", "food"],
    "cake": ["cupcake", "frosting", "icing", "dessert", "food"],
    "banana": ["banana peel", "fruit", "food"],
    "orange": ["fruit", "tangerine", "clementine", "food"],
    "apple": ["fruit", "apple logo", "food"],
    "broccoli": ["vegetable", "food"],
    "sandwich": ["bun", "bread", "food", "sub", "burger", "hamburger"],
    "donut": ["pastry", "dessert", "food"],
    "carrot": ["vegetable", "food"],
    "hot dog": ["hotdog bun", "bun", "sausage", "food"],
    "bread": ["bun", "bagel", "toast", "sandwich", "roll", "crust", "food"],
    # plants and misc
    "plant": ["flower", "flower pot", "pot", "planter", "bush", "vine", "leaves", "leaf"],
    "flower": ["plant", "bouquet", "flower pot", "planter", "blossom"],
    "statue": ["figurine", "monument"],
    "bucket": ["pail", "container", "trash can"],
    "ladder": ["steps"],
}
# animal-specific parts and look-alikes, added to the shared animal blockers above
for animal, extra in {
    "dog": FUR_PARTS + ["stuffed dog"],
    "cat": FUR_PARTS,
    "horse": ["mane", "hoof", "horse hoof", "saddle", "reins", "bridle", "zebra"],
    "bird": ["beak", "feathers", "feather", "wing", "chicken", "hen", "rooster"],
    "elephant": ["tusk", "trunk"],
    "giraffe": ["mane"],
    "zebra": ["mane", "horse"],
    "bear": FUR_PARTS + ["teddy bear", "stuffed bear"],
    "cow": ["horns", "horn", "hoof", "ox", "goat"],
    "sheep": ["wool", "goat", "ram"],
}.items():
    BLOCKERS[animal] = BLOCKERS[animal] + extra


def _forms(name: str) -> set[str]:
    return {name, name + "s", name + "es"}


def name_index() -> tuple[dict[str, str], dict[str, set[str]]]:
    """(name -> object it means, object -> all names that block a "no" for it)."""
    means = {}
    for obj, names in SYNONYMS.items():
        for n in names:
            for f in _forms(n):
                if f in means and means[f] != obj:
                    raise ValueError(f"name {f!r} is a synonym of both {means[f]!r} and {obj!r}")
                means[f] = obj
    blocks = {obj: set().union(*(_forms(n) for n in SYNONYMS[obj] + BLOCKERS.get(obj, [])))
              for obj in SYNONYMS}
    return means, blocks


def scene_objects(graph: dict) -> list[dict]:
    """Objects of one scene graph with lower-cased, stripped names, in scene-graph order."""
    return [{**o, "name": o["name"].strip().lower()} for o in graph["objects"].values()]


def coverage_report(scene_graphs: dict) -> None:
    means, blocks = name_index()
    present_imgs = Counter()
    blocked_imgs = Counter()
    name_hits = {obj: Counter() for obj in SYNONYMS}
    all_names = Counter()
    eligible = 0
    for g in scene_graphs.values():
        names = {o["name"] for o in scene_objects(g)}
        all_names.update(names)
        present = {means[n] for n in names if n in means}
        eligible += len(present) >= 3
        for obj in SYNONYMS:
            if obj in present:
                present_imgs[obj] += 1
                name_hits[obj].update(n for n in names if means.get(n) == obj)
            elif names & blocks[obj]:
                blocked_imgs[obj] += 1
    n = len(scene_graphs)
    print(f"{len(SYNONYMS)} vocabulary objects; {eligible:,}/{n:,} images have >= 3 of them\n")
    print(f"{'object':16} {'present':>8} {'blocked':>8}  synonyms seen (images)")
    for obj in sorted(SYNONYMS, key=lambda o: -present_imgs[o]):
        seen = ", ".join(f"{k}({v})" for k, v in name_hits[obj].most_common(6))
        print(f"{obj:16} {present_imgs[obj]:8,} {blocked_imgs[obj]:8,}  {seen}")
    listed = set(means) | set().union(*blocks.values())
    unused = [(k, v) for k, v in all_names.most_common() if k not in listed and v >= 40]
    print(f"\nnames in >= 40 images that the vocabulary does not mention ({len(unused)}):")
    print(", ".join(f"{k}({v})" for k, v in unused))
    never = sorted(n for obj, names in SYNONYMS.items() for n in names if n not in all_names)
    print(f"\nlisted synonyms never seen in GQA val (harmless, kept for safety): {', '.join(never)}")


if __name__ == "__main__":
    root = os.environ.get("CMPE297_DATA", os.path.expanduser("~/cmpe297/data"))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "gqa/sceneGraphs/val_sceneGraphs.json")
    coverage_report(json.load(open(path)))
