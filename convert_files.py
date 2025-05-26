from pathlib import Path

# Redefine all conversion logic due to reset
posts_dir = Path("_posts")
posts_dir.mkdir(parents=True, exist_ok=True)

# Rebuild content from extracted markdown
conversion_map = {
    "README.md": ("2020-01-01-remote-sensing-tutorials.md", "Remote Sensing Tutorials"),
    "Cirrus_Correction_All_Bands_L8.md": ("2020-01-02-cirrus-correction.md", "Cirrus Correction of Landsat 8"),
    "Mapping_potato_late_blight_from_UAV-based_multispectral_imagery.md": ("2020-01-03-potato-blight-part1.md", "Potato Late Blight – Part 1"),
    "Multispectral_imagery_classification_using_pre-trained_models.md": ("2020-01-04-potato-blight-part2.md", "Potato Late Blight – Part 2")
}

def convert_to_jekyll_post(input_file, output_name, title):
    input_path = Path("./") / input_file
    output_path = posts_dir / output_name
    date_str = output_name.split("-")[0:3]
    date = "-".join(date_str)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    front_matter = f"""---
layout: post
title: "{title}"
date: {date}
---
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(front_matter + content)

# Perform conversion
for original, (filename, title) in conversion_map.items():
    convert_to_jekyll_post(original, filename, title)

posts_dir.name