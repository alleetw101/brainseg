# Brain Seg

![Segmentation Comparison](assets/SegmentationComparison.jpg)

A brain segmentation tool using machine learning

Under development

[Colab](https://colab.research.google.com/drive/1cTKJ7WxrV41UKfQ2oieYbxdz3m_Lvga2?usp=sharing)

This software has NOT been approved for medical analysis/diagnosis. Please use other
approved software for medical use.

# Models
SavedModels for use can be found at [Google Drive](https://drive.google.com/drive/folders/1DuKPhzKcxWUNXYxYbSNw8zydiAG7a8qw?usp=sharing).

Models were trained using the Mindboggle-101 data:

Klein, Arno, 2016, "Mindboggle-101 manually labeled individual brains", 
https://doi.org/10.7910/DVN/HMQKCK, Harvard Dataverse, V2

A Klein, SS Ghosh, FS Bao, J Giard, Y Hame, E Stavsky, N Lee, B Rossa, M Reuter, EC Neto, A Keshavan. 2017. 
Mindboggling morphometry of human brains. 
PLoS Computational Biology 13(3): e1005350. doi:10.1371/journal.pcbi.1005350 
(bioRxiv 2016 preprint)

# Options
    positional arguments:
      filepath              Path of file to segment
    
    optional arguments:
      -h, --help            show this help message and exit
      -s SEGMENT, --segment Segments using provided SavedModel (Directory path required)
      --segment-precision   Float32 or float64 {32,64}
      --output OUTPUT, -o   Output filepath
      --output-channels     Number of channel in output image {1,3}


# Usage Examples
```bash
# Segments a sample nifti scan
$ brainseg sample.nii.gz -s sampleSavedModel -o output.nii.gz
```

# Model Training Graphs

SavedModel_210303
![accuracy_210303](assets/accuracy_210303_235240.png)
![loss_210303](assets/loss_210303_235240.png)
