# Binary-Segmentation
Performs semi-automatic binary segmentation based on SLIC superpixels and graph-cuts
Given an image and sparse markings for foreground and background
Calculate SLIC over image
Calculate color histograms for all superpixels
Calculate color histograms for FG and BG
Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
Run a graph-cut algorithm to get the final segmentation


"Wow factor" bonus :(Video uploaded for this part)
Lets the user draw the markings for every interaction step (mouse click, drag, etc.)
1. Recalculates only the FG-BG histograms,
2. Construct the graph and get a segmentation from the max-flow graph-cut,
3. Show the result immediately to the user.
