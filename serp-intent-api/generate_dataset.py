import csv
import random
import argparse
import os
import locale

CITY_PACKS = {
    "NL": ["Amsterdam", "Rotterdam", "Utrecht", "Eindhoven", "Arnhem", "Groningen", "Den Haag", "Nijmegen"],
    "DE": ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt", "Stuttgart", "Düsseldorf", "Leipzig"],
    "FR": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Bordeaux"],
    "GB": ["London", "Manchester", "Birmingham", "Leeds", "Liverpool", "Bristol", "Sheffield", "Glasgow"],
    "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Diego", "Dallas"],
    "GLOBAL": [
        "Amsterdam", "Berlin", "Paris", "London", "New York", "Madrid", "Rome", "Vienna",
        "Stockholm", "Copenhagen", "Dublin", "Lisbon", "Prague", "Warsaw", "Athens", "Brussels"
    ]
}

products = [
    "iphone 15",
    "airpods pro",
    "robot vacuum",
    "gaming laptop",
    "coffee machine",
    "running shoes",
    "protein powder",
    "office chair",
    "wireless headphones",
    "electric bike",
]

brands = [
    "facebook",
    "paypal",
    "amazon",
    "netflix",
    "spotify",
    "instagram",
    "linkedin",
    "x",
]

tasks = [
    "cook rice",
    "lose weight",
    "build muscle",
    "start a business",
    "learn python",
    "change a tire",
    "edit videos",
    "invest in stocks",
]

def detect_country_code() -> str:
    loc = None
    try:
        loc = locale.getdefaultlocale()[0]  # e.g. "nl_NL"
    except Exception:
        loc = None

    if not loc:
        loc = os.environ.get("LANG", "")

    loc = (loc or "").strip()

    if "_" in loc:
        tail = loc.split("_", 1)[1]
        country = tail.split(".", 1)[0].upper()
        return country

    return "GLOBAL"

def load_cities(country: str, cities_file: str | None) -> list[str]:
    if cities_file:
        with open(cities_file, "r", encoding="utf-8") as f:
            cities = [line.strip() for line in f if line.strip()]
        return cities if cities else CITY_PACKS["GLOBAL"]

    country = (country or "").upper()
    return CITY_PACKS.get(country, CITY_PACKS["GLOBAL"])

def generate_rows(cities: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []

    # Transactional (often: shopping ads / product listings)
    for p in products:
        rows.append([f"buy {p}", "transactional", "shopping_ads,product_listings"])
        rows.append([f"{p} price", "transactional", "shopping_ads"])
        rows.append([f"{p} discount", "transactional", "shopping_ads"])
        rows.append([f"order {p} online", "transactional", "shopping_ads,product_listings"])

    # Commercial (often: reviews / PAA)
    for p in products:
        rows.append([f"best {p}", "commercial", "reviews,people_also_ask"])
        rows.append([f"{p} review", "commercial", "reviews"])
        rows.append([f"{p} comparison", "commercial", "people_also_ask"])
        rows.append([f"{p} vs alternatives", "commercial", "people_also_ask"])

    # Informational (often: featured snippet / PAA)
    for t in tasks:
        rows.append([f"how to {t}", "informational", "featured_snippet,people_also_ask"])
        rows.append([f"what is {t}", "informational", "featured_snippet"])
        rows.append([f"{t} guide", "informational", "people_also_ask"])

    # Navigational (often: sitelinks / knowledge panel)
    for b in brands:
        rows.append([f"{b} login", "navigational", "sitelinks"])
        rows.append([f"{b} customer service", "navigational", "sitelinks"])
        rows.append([f"contact {b}", "navigational", "sitelinks"])

    # Local (often: local pack / map)
    for p in products:
        for c in cities:
            rows.append([f"{p} in {c}", "local", "local_pack"])
            rows.append([f"buy {p} in {c}", "local", "local_pack,shopping_ads"])

        rows.append([f"{p} near me", "local", "local_pack"])
        rows.append([f"{p} nearby", "local", "local_pack"])
        rows.append([f"{p} in my area", "local", "local_pack"])

    random.shuffle(rows)
    return rows

def main():
    parser = argparse.ArgumentParser(description="Generate a labeled intent dataset (query,intent).")
    parser.add_argument("--country", type=str, default=None, help="Country code like NL, DE, FR, GB, US. Default: auto-detect from OS locale.")
    parser.add_argument("--cities-file", type=str, default=None, help="Path to a text file with one city per line.")
    parser.add_argument("--out", type=str, default="dataset.csv", help="Output CSV filename.")
    args = parser.parse_args()

    country = args.country or detect_country_code()
    cities = load_cities(country, args.cities_file)

    rows = generate_rows(cities)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "intent", "serp_features"])
        writer.writerows(rows)

    print(f"Detected/selected country: {country}")
    print(f"Cities used: {len(cities)}")
    print(f"Generated rows: {len(rows)}")
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()

def generate_rows(cities: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []

    # Transactional (often: shopping ads / product listings)
    for p in products:
        rows.append([f"buy {p}", "transactional", "shopping_ads,product_listings"])
        rows.append([f"{p} price", "transactional", "shopping_ads"])
        rows.append([f"{p} discount", "transactional", "shopping_ads"])
        rows.append([f"order {p} online", "transactional", "shopping_ads,product_listings"])

    # Commercial (often: reviews / PAA)
    for p in products:
        rows.append([f"best {p}", "commercial", "reviews,people_also_ask"])
        rows.append([f"{p} review", "commercial", "reviews"])
        rows.append([f"{p} comparison", "commercial", "people_also_ask"])
        rows.append([f"{p} vs alternatives", "commercial", "people_also_ask"])

    # Informational (often: featured snippet / PAA)
    for t in tasks:
        rows.append([f"how to {t}", "informational", "featured_snippet,people_also_ask"])
        rows.append([f"what is {t}", "informational", "featured_snippet"])
        rows.append([f"{t} guide", "informational", "people_also_ask"])

    # Navigational (often: sitelinks / knowledge panel)
    for b in brands:
        rows.append([f"{b} login", "navigational", "sitelinks"])
        rows.append([f"{b} customer service", "navigational", "sitelinks"])
        rows.append([f"contact {b}", "navigational", "sitelinks"])

    # Local (often: local pack / map)
    for p in products:
        for c in cities:
            rows.append([f"{p} in {c}", "local", "local_pack"])
            rows.append([f"buy {p} in {c}", "local", "local_pack,shopping_ads"])

        rows.append([f"{p} near me", "local", "local_pack"])
        rows.append([f"{p} nearby", "local", "local_pack"])
        rows.append([f"{p} in my area", "local", "local_pack"])

    random.shuffle(rows)
    return rows
