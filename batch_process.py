#!/usr/bin/env python3
"""Batch process multiple images to STL relief models."""

import os
import sys
from pathlib import Path
from PIL import Image
from glob import glob
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from relief import ReliefGenerator

def process_image(image_path, output_dir=None, **kwargs):
    """Process a single image to STL."""
    if output_dir is None:
        output_dir = Path(image_path).parent
    else:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    output_path = output_dir / f"{Path(image_path).stem}_relief.stl"
    
    print(f"Processing: {image_path} -> {output_path}")
    
    img = Image.open(image_path).convert('RGB')
    
    gen = ReliefGenerator()
    gen.load_image_from_pil(img, max_size=kwargs.get('max_size', 100))
    gen.create_depth_map()
    gen.create_relief_mesh(
        model_width=kwargs.get('width', 80.0),
        model_thickness=kwargs.get('thickness', 5.0),
        base_thickness=kwargs.get('base', 2.0)
    )
    
    gen.save_stl(str(output_path))
    
    info = gen.get_mesh_info()
    print(f"  -> {info.get('triangle_count', info.get('total_triangles', 'N/A')):,} triangles, {output_path.stat().st_size:,} bytes")
    
    return output_path

def batch_process(input_pattern, output_dir=None, **kwargs):
    """Process multiple images matching a glob pattern."""
    
    images = glob(input_pattern)
    if not images:
        print(f"No images found matching: {input_pattern}")
        return []
    
    print(f"Found {len(images)} images to process")
    
    results = []
    for img_path in images:
        try:
            result = process_image(img_path, output_dir, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nProcessed {len(results)}/{len(images)} images successfully")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert images to STL relief models")
    parser.add_argument("input", help="Input image file or glob pattern (use '*.png' for batch)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--width", type=float, default=80.0, help="Model width in mm (default: 80)")
    parser.add_argument("--thickness", type=float, default=5.0, help="Relief thickness in mm (default: 5)")
    parser.add_argument("--base", type=float, default=2.0, help="Base thickness in mm (default: 2)")
    parser.add_argument("--max-size", type=int, default=100, help="Max image dimension (default: 100)")

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        print("\nError: Please provide an input file or pattern.")
        sys.exit(1)

    batch_process(args.input, args.output,
        width=args.width,
        thickness=args.thickness,
        base=args.base,
        max_size=args.max_size)
