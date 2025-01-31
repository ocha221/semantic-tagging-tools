import numpy as np
import faiss
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install
from rich import print
import sys
from typing import List, Dict
import json
import time
from datetime import datetime
import os
from mistralai import Mistral
import argparse  

install()
console = Console()

EMBEDDING_CALLS = 0
LLM_QUERY_CALLS = 0


def init_mistral(api_key: str) -> Mistral: 
    try:
        return Mistral(api_key)
    except Exception as e:
        console.print(f"[red]Couldnt init mistral: {str(e)}[/red]")
        sys.exit(1)


def generate_embeddings(mimi: Mistral, tags: List[str]) -> np.ndarray:
    global EMBEDDING_CALLS
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Generating embeddings...", total=1)
        try:
            EMBEDDING_CALLS += 1
            response = mimi.embeddings.create(
                model="mistral-embed",
                inputs=tags,
            )
            embeddings = np.array([data.embedding for data in response.data]).astype(
                "float32"
            )
            faiss.normalize_L2(embeddings)
            progress.update(task, completed=1)
            return embeddings
        except Exception as e:
            console.print(f"[red]Error generating embeddings: {str(e)}[/red]")
            sys.exit(1)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    with console.status("[bold green]Building FAISS index..."):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index


def log_response(prompt: str, response: str, batch_info: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = batch_info.split(",", 1)
    tag = parts[0].replace("Tag:", "").strip()
    candidates = parts[1].replace("Candidates:", "").strip()

    candidate_list = [c.strip() for c in candidates.strip("[]").split(",")]
    formatted_candidates = ", ".join(candidate_list)

    with open("llmout.log", "a") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"{timestamp}\n")
        f.write(f"Input Tag: {tag}\n")
        f.write(f"Candidates: {formatted_candidates}\n")
        f.write(f"LLM Response:\n{response}\n")


def log_debug(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("debug.log", "a") as f:
        f.write(f"{timestamp}: {message}\n")


def get_synonyms(mimi: Mistral, tag: str, candidate_batch: List[str]) -> List[str]:
    """Process a single tag and one batch of its candidates."""
    global LLM_QUERY_CALLS
    try:
        prompt = """Your task is to identify ONLY exact synonyms and DIRECT PARENT categories for the input word. 
IMPORTANT: Never include specific types/subtypes of the input word!
DECISION RULES:
1. SYNONYMS: Include ONLY if:
   - Words mean EXACTLY the same thing (like "car" = "automobile")
   - Can substitute in ANY context with NO change in meaning
   
2. PARENT CATEGORIES: Include ONLY if:
   - Parent is MORE GENERAL than input word
   - Can say "X is a type of Y" but NOT "Y is a type of X"
   
CRITICAL RELATIONSHIP DIRECTION:
VALID "metal" -> "material" (VALID: metal is a type of material)
INVALID "metal" -> "brass" (INVALID: brass is a type of metal - wrong direction!)
INVALID "metal" -> "bronze" (INVALID: bronze is a type of metal - wrong direction!)

MORE EXAMPLES:
"dog" -> "animal" (VALID: parent category)
"dog" -> "poodle" (INVALID: poodle is a type of dog - wrong direction!)

"vehicle" -> "car" (INVALID: car is a type of vehicle - wrong direction!)
"vehicle" -> "transport" (VALID: parent category)

AUTOMATIC EXCLUSION:
- Any subtypes or specific varieties of the input word
- Related terms that aren't strictly synonyms
- Specific examples of the input category

EXAMPLE OUTPUTS:
Input: "metal"
Candidates: "brass, bronze, gold, material, substance"
Valid output: {"metal": ["material", "substance"]}  # only parent categories, NO subtypes

Input: "dog"
Candidates: "poodle, animal, pet, mammal, canine"
Valid output: {"dog": ["animal", "mammal"]}  # only parent categories

Only use the words provided in the candidates list. Do NOT add any new words.
Reply with EXACTLY: {
"word1": ["syn1", "syn2"], ...
"word2": ["syn3", "syn4"], ...
}"""

        user_prompt = f'\nWord: "{tag}"\nCandidates: {", ".join(candidate_batch)}\n'

        LLM_QUERY_CALLS += 1
        response = mimi.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.05,
        )

        batch_info = f"Tag: {tag}, Candidates: {candidate_batch}"
        log_response("", response.choices[0].message.content, batch_info)

        result = json.loads(response.choices[0].message.content)
        return result.get(tag, [])
    except Exception as e:
        console.print(f"[yellow]Error processing tag {tag}: {str(e)}[/yellow]")
        log_response(prompt, f"ERROR: {str(e)}", f"Failed tag: {tag}")
        return []


def process_tag_candidates(
    tag: str, candidates: List[str], batch_size: int = 10
) -> List[List[str]]:
    """Split candidates into smaller batches."""
    return [
        candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)
    ]


def save_synonym_mapping(
    mapping: Dict[str, List[str]], filepath: str = "synonym_mapping.json"
):
    with open(filepath, "w") as f:
        json.dump(mapping, f, indent=2)
    console.print(f"[green]Saved synonym mapping to {filepath}[/green]")


def load_synonym_mapping(
    filepath: str = "synonym_mapping.json",
) -> Dict[str, List[str]]:
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def show_menu(synonym_mapping: Dict[str, List[str]]) -> None:
    """Display and handle the interactive menu for tag operations."""
    while True:
        menu = Table(show_header=False, box=None)
        menu.add_row("[cyan]1.[/cyan] Try example 1 (car, red, new)")
        menu.add_row("[cyan]2.[/cyan] Try example 2 (laptop, wireless, professional)")
        menu.add_row("[cyan]3.[/cyan] Try example 3 (metal, shiny, industrial)")
        menu.add_row("[cyan]4.[/cyan] Enter custom tags")
        menu.add_row("[cyan]5.[/cyan] View all synonyms")
        menu.add_row("[cyan]6.[/cyan] Exit")

        console.print("\n[bold]Menu:[/bold]")
        console.print(menu)

        choice = console.input("\n[bold cyan]Enter your choice (1-6): [/bold cyan]")

        if choice == "1":
            example_tags = ["car", "red", "new"]
            result = augment_tags(example_tags, synonym_mapping)
            console.print(
                Panel(
                    f"Original: {example_tags}\nAugmented: {result}",
                    title="Result",
                    border_style="green",
                )
            )

        elif choice == "2":
            example_tags = ["laptop", "wireless", "professional"]
            result = augment_tags(example_tags, synonym_mapping)
            console.print(
                Panel(
                    f"Original: {example_tags}\nAugmented: {result}",
                    title="Result",
                    border_style="green",
                )
            )

        elif choice == "3":
            example_tags = ["metal", "shiny", "industrial"]
            result = augment_tags(example_tags, synonym_mapping)
            console.print(
                Panel(
                    f"Original: {example_tags}\nAugmented: {result}",
                    title="Result",
                    border_style="green",
                )
            )

        elif choice == "4":
            tags_input = console.input(
                "[bold cyan]Enter tags (comma-separated): [/bold cyan]"
            )
            custom_tags = [tag.strip() for tag in tags_input.split(",")]
            result = augment_tags(custom_tags, synonym_mapping)
            console.print(
                Panel(
                    f"Original: {custom_tags}\nAugmented: {result}",
                    title="Result",
                    border_style="green",
                )
            )

        elif choice == "5":
            syn_table = Table(show_header=True)
            syn_table.add_column("Tag", style="cyan")
            syn_table.add_column("Synonyms", style="green")
            for tag, syns in sorted(synonym_mapping.items()):
                syn_table.add_row(tag, ", ".join(syns))
            console.print(syn_table)

        elif choice == "6":
            console.print("[bold green]Goodbye![/bold green]")
            break

        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")


def augment_tags(obj_tags: List[str], synonym_map: Dict[str, List[str]]) -> List[str]:
    try:
        augmented = set(obj_tags)
        for tag in obj_tags:
            augmented.update(synonym_map.get(tag, []))
        return list(augmented)
    except Exception as e:
        console.print(f"[red]Error during tag augmentation: {str(e)}[/red]")
        return obj_tags
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of synonym mapping",
    )
    args = parser.parse_args()

    console.print(Panel.fit("Tag merger", style="bold magenta"))

    if not args.regenerate:
        existing_mapping = load_synonym_mapping()
        if existing_mapping:
            console.print("[green]Loaded existing synonym mapping[/green]")
            show_menu(existing_mapping)
            return

    API_KEY = ""  # * your API key
    mimi = init_mistral(API_KEY)

    # * you can programatically fill this up using pre-existing tags from your dataset
    # * these are generally too few tags to be useful
    all_tags = [
       
        "car",
        "automobile",
        "vehicle",
        "sedan",
        "suv",
        "truck",
        "motorcycle",
        "bicycle",
        "bike",
        "coupe",
        "hatchback",
        "van",
        "minivan",
        "convertible",
        "roadster",
        "pickup",
        "lorry",
        "jeep",
        "electric vehicle",
        "EV",
        "hybrid",
        "gasoline",
        "diesel",
        "electric",
        "moped",
        "scooter",
        "skateboard",
        "bus",
        "train",
        "tram",
        "subway",
        "metro",
        "airplane",
        "aircraft",
        "helicopter",
        "boat",
        "ship",
        "yacht",
        "vessel",
        "off-road",
        "4x4",
        "4wd",
        "awd",
        "fwd",
        "rwd",
        "dune buggy",
        "atv",
        "go-kart",
        "racecar",
        "formula 1",
        "rally car",
        # * Colors
        "red",
        "blue",
        "green",
        "white",
        "black",
        "silver",
        "grey",
        "gray",
        "yellow",
        "purple",
        "orange",
        "pink",
        "brown",
        "beige",
        "gold",
        "cyan",
        "magenta",
        "lime",
        "teal",
        "indigo",
        "violet",
        "maroon",
        "navy",
        "olive",
        "turquoise",
        "lavender",
        "coral",
        "crimson",
        "aqua",
        "metallic",
        "pearlescent",
        "iridescent",
        "chromatic",
        "monochromatic",
    
        "Korean",
        "Japanese",
        "American",
        "German",
        "Italian",
        "Chinese",
        "British",
        "French",
        "Swedish",
        "Spanish",
        "Mexican",
        "Canadian",
        "Australian",
        "Indian",
        "Russian",
        "Brazilian",
        "European",
        "Asian",
        "African",
        "Latin American",
       
        "circuit",
        "electronics",
        "pcb",
        "printed circuit board",
        "motherboard",
        "cpu",
        "central processing unit",
        "gpu",
        "graphics processing unit",
        "ram",
        "random access memory",
        "ssd",
        "solid state drive",
        "hdd",
        "hard disk drive",
        "memory",
        "storage",
        "screen",
        "monitor",
        "display",
        "lcd",
        "liquid crystal display",
        "led",
        "light emitting diode",
        "oled",
        "organic light emitting diode",
        "touchscreen",
        "panel",
        "touch panel",
        "digitizer",
        "keyboard",
        "mouse",
        "trackpad",
        "laptop",
        "desktop",
        "tablet",
        "smartphone",
        "device",
        "gadget",
        "gizmo",
        "mobile phone",
        "cell phone",
        "smartwatch",
        "wearable",
        "headphones",
        "earphones",
        "earbuds",
        "speakers",
        "audio",
        "video",
        "camera",
        "lens",
        "sensor",
        "microchip",
        "semiconductor",
        "transistor",
        "capacitor",
        "resistor",
        "inductor",
        "diode",
        "integrated circuit",
        "IC",
        "SoC",
        "system on chip",
        "firmware",
        "software",
        "hardware",
        "driver",
        "operating system",
        "OS",
        "android",
        "iOS",
        "windows",
        "macOS",
        "linux",
        "unix",
        "programming",
        "coding",
        "algorithm",
        "data",
        "database",
        "cloud",
        "server",
        "cybersecurity",
        "artificial intelligence",
        "AI",
        "machine learning",
        "deep learning",
        "neural network",
        "robotics",
        "automation",
       
        "metal",
        "plastic",
        "glass",
        "wood",
        "steel",
        "stainless steel",
        "aluminum",
        "copper",
        "silicon",
        "carbon fiber",
        "fiberglass",
        "rubber",
        "leather",
        "fabric",
        "textile",
        "ceramic",
        "composite",
        "alloy",
        "titanium",
        "brass",
        "bronze",
        "polymer",
        "vinyl",
    
        "new",
        "used",
        "vintage",
        "modern",
        "retro",
        "classic",
        "contemporary",
        "antique",
        "old",
        "aged",
        "refurbished",
        "reconditioned",
        "second-hand",
        "pre-owned",
        "broken",
        "damaged",
        "faulty",
        "defective",
        "malfunctioning",
        "repaired",
        "restored",
        "maintained",
        "serviced",
        "upgraded",
        "modified",
        "customized",
        "personalized",
        "stock",
        "default",
        "original",
        "oem",
        "aftermarket",
      
        "indoor",
        "outdoor",
        "inside",
        "outside",
        "exterior",
        "interior",
        "street",
        "road",
        "highway",
        "freeway",
        "motorway",
        "garage",
        "workshop",
        "laboratory",
        "office",
        "home",
        "house",
        "apartment",
        "building",
        "factory",
        "warehouse",
        "store",
        "shop",
        "park",
        "nature",
        "city",
        "urban",
        "rural",
        "suburban",
  
        "clean",
        "dirty",
        "dusty",
        "filthy",
        "spotless",
        "pristine",
        "grimy",
        "soiled",
        "rusty",
        "corroded",
        "oxidized",
        "tarnished",
        "shiny",
        "glossy",
        "lustrous",
        "polished",
        "reflective",
        "matte",
        "dull",
        "flat",
        "textured",
        "rough",
        "smooth",
        "sleek",
        "soft",
        "hard",
        "rigid",
        "flexible",
        "delicate",
        "sturdy",
        "rugged",
        "durable",
        "fragile",
       
        "big",
        "small",
        "medium",
        "compact",
        "large",
        "tiny",
        "miniature",
        "micro",
        "nano",
        "huge",
        "massive",
        "enormous",
        "gigantic",
        "giant",
        "colossal",
        "bulky",
        "heavy",
        "light",
        "lightweight",
        "portable",
        "mini",
      
        "expensive",
        "cheap",
        "affordable",
        "budget",
        "luxury",
        "premium",
        "high-end",
        "low-end",
        "mid-range",
        "value",
        "bargain",
        "discount",
        "sale",
        "clearance",
        "pricey",
        "costly",
        "economical",
        "inexpensive",
   
        "fast",
        "slow",
        "quick",
        "rapid",
        "swift",
        "sluggish",
        "efficient",
        "inefficient",
        "powerful",
        "weak",
        "strong",
        "feeble",
        "robust",
        "sturdy",
        "reliable",
        "unreliable",
        "stable",
        "unstable",
        "consistent",
        "inconsistent",
        "durable",
        "fragile",
        "precise",
        "accurate",
        "inaccurate",
        "noisy",
        "quiet",
        "silent",
        "vibrant",
        "faded",
      
        "hot",
        "cold",
        "warm",
        "cool",
        "freezing",
        "boiling",
        "lukewarm",
        "temperate",
        "temperature",
        "climate",
        "weather",
        "humid",
        "dry",
        "wet",
        "damp",
        "moist",
        "rainy",
        "sunny",
        "cloudy",
        "foggy",
        "windy",
        "stormy",
        "seasonal",
        "tropical",
        "arctic",
        "desert",
   
        "professional",
        "amateur",
        "enthusiast",
        "beginner",
        "expert",
        "novice",
        "industrial",
        "commercial",
        "residential",
        "personal",
        "private",
        "public",
        "business",
        "enterprise",
        "domestic",
        "hobby",
        "diy",
        "do it yourself",
      
        "wired",
        "wireless",
        "cordless",
        "bluetooth",
        "wifi",
        "wi-fi",
        "cellular",
        "network",
        "internet",
        "connected",
        "online",
        "offline",
        "ethernet",
        "usb",
        "hdmi",
        "vga",
        "dvi",
        "displayport",
        "infrared",
        "rf",
        "radio frequency",
        "nfc",
        "near field communication",
 
        "branded",
        "unbranded",
        "generic",
        "genuine",
        "fake",
        "counterfeit",
        "authentic",
        "replica",
        "original",
        "imitation",
        "knock-off",
        "trademark",
        "copyright",
        "patent",
      
        "artificial intelligence",
        "AI",
        "machine learning",
        "ML",
        "deep learning",
        "neural network",
        "neural net",
        "computer vision",
        "natural language processing",
        "NLP",
        "speech recognition",
        "voice recognition",
        "machine translation",
        "image recognition",
        "object detection",
        "pattern recognition",
        "data mining",
        "big data",
        "data science",
        "data analytics",
        "predictive analytics",
        "algorithm",
        "model",
        "training",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "transfer learning",
        "federated learning",
        "generative AI",
        "GAN",
        "generative adversarial network",
        "VAE",
        "variational autoencoder",
        "robotics",
        "automation",
        "autonomous",
        "self-driving",
        "smart",
        "intelligent",
        "cognitive computing",
        "expert system",
        "knowledge base",
        "inference engine",
        "chatbot",
        "virtual assistant",
        "conversational AI",
        "bot",
        "cloud computing",
        "edge computing",
        "quantum computing",
        "cybersecurity",
        "encryption",
        "blockchain",
        "decentralized",
        "IoT",
        "internet of things",
        "smart home",
        "smart city",
        "augmented reality",
        "AR",
        "virtual reality",
        "VR",
        "mixed reality",
        "MR",
        "extended reality",
        "XR",
        "algorithm",
        "model",
        "dataset",
        "training data",
        "test data",
        "validation data",
        "feature engineering",
        "dimensionality reduction",
        "clustering",
        "classification",
        "regression",
        "bias",
        "fairness",
        "explainable AI",
        "XAI",
        "interpretable AI",
        "GPU",
        "TPU",
        "tensor processing unit",
        "parallel processing",
        "distributed computing",
        "API",
        "application programming interface",
        "SDK",
        "software development kit",
        "open source",
        "proprietary",
        "cloud-based",
        "on-premise",
        "ethical AI",
        "responsible AI",
        "AI safety",
        "AI alignment",
        "handheld",
        "portable",
        "wearable",
        "digital",
        "analog",
        "automatic",
        "manual",
        "electric",
        "electronic",
        "mechanical",
        "hand-crafted",
        "mass-produced",
        "modular",
        "integrated",
        "standalone",
        "waterproof",
        "water-resistant",
        "dustproof",
        "shockproof",
        "eco-friendly",
        "sustainable",
        "recycled",
        "biodegradable",
        "minimalist",
        "ornate",
        "decorative",
        "functional",
        "ergonomic",
        "aesthetic",
        "utilitarian",
        "stylish",
        "trendy",
        "fashionable",
        "unusual",
        "unique",
        "rare",
        "common",
        "popular",
        "mainstream",
        "niche",
        "cutting-edge",
        "innovative",
        "state-of-the-art",
        "obsolete",
        "outdated",
        "manual",
    ]

    # * embedding
    console.print("\n[bold]Step 1: Generating Embeddings[/bold]")
    if not os.path.exists("embeddings.npy"):
        embeddings = generate_embeddings(mimi, all_tags)
        np.save("embeddings.npy", embeddings)
    else:
        embeddings = np.load("embeddings.npy")

    # *vec  index
    console.print("\n[bold]Step 2: Building Search Index[/bold]")
    index = build_faiss_index(embeddings)

    # *llm jobs
    console.print("\n[bold]Step 3: Finding Synonyms[/bold]")
    synonym_mapping = {}

    total_tasks = 0
    tag_candidates_map = {}

    log_debug(f"[dim]Total tags to process: {len(all_tags)}[/dim]")

    for idx, tag in enumerate(all_tags):
        query_embedding = embeddings[idx].reshape(1, -1)
        distances, indices = index.search(query_embedding, 11) #! change this to get more neighbours
        candidates = [
            all_tags[i]
            for i, dist in zip(indices[0], distances[0])
            if all_tags[i] != tag and dist < 0.65
        ][:10] #! and change this to include more results 

        if candidates:
            batches = process_tag_candidates(tag, candidates, batch_size=20) #! check readme
            tag_candidates_map[tag] = batches
            total_tasks += len(batches)

            log_debug(
                f"[dim]Tag '{tag}' has {len(candidates)} candidates in {len(batches)} batches[/dim]"
            )

    log_debug(f"[dim]Total tasks to process: {total_tasks}[/dim]")
    #if paying it should be pretty straightforward to make this run multiple jobs at once
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Processing tags and candidates...", total=total_tasks
        )

        for tag, candidate_batches in tag_candidates_map.items(): 
            tag_synonyms = set()

            log_debug(f"[dim]Starting processing for tag: {tag}[/dim]")

            for batch_idx, candidate_batch in enumerate(candidate_batches, 1):
                if not candidate_batch:  # * skip empty
                    continue

                batch_synonyms = get_synonyms(mimi, tag, candidate_batch)
                tag_synonyms.update(batch_synonyms)

                progress.update(task, advance=1)
                log_debug(
                    f"[dim]Processed {tag} - batch {batch_idx}/{len(candidate_batches)} - Found {len(batch_synonyms)} synonyms[/dim]"
                )

                time.sleep(1.8)  # * free api has a 1rps limit (not complaining!)

            if tag_synonyms:
                synonym_mapping[tag] = list(tag_synonyms)

    log_debug(f"[dim]Final synonym mapping contains {len(synonym_mapping)} tags[/dim]")

    # *stats
    console.print("\n[bold]Synonym Mapping Statistics:[/bold]")
    stats_table = Table(show_header=True, header_style="bold green")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right")

    total_synonyms = sum(len(syns) for syns in synonym_mapping.values())
    avg_synonyms = total_synonyms / len(all_tags)

    stats_table.add_row("Total Tags", str(len(all_tags)))
    stats_table.add_row("Total Synonyms Found", str(total_synonyms))
    stats_table.add_row("Average Synonyms per Tag", f"{avg_synonyms:.2f}")
    stats_table.add_row("Total Embedding API Calls", str(EMBEDDING_CALLS))
    stats_table.add_row("Total LLM Query API Calls", str(LLM_QUERY_CALLS))

    console.print(stats_table)
    show_menu(synonym_mapping)


if __name__ == "__main__":
    main()
