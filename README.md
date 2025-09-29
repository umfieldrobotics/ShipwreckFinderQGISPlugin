# Shipwreck Finder QGIS Plugin

![QGIS_Main](https://github.com/user-attachments/assets/946e6332-d93b-4585-94af-745456292167)

This [QGIS](https://www.qgis.org/en/site/) plugin identifies potential shipwreck sites from multibeam sonar data. Raster layers are passed through a deep learning segmentation model to produce a binary segmentation mask that clearly identifies the shipwreck site, and bounding boxes can be extracted from these segmentation masks if desired. Four different models are available for use, each trained with a unique backbone, and each with their own strengths and weaknesses.

This repo accompanies the paper "ShipwreckFinder: A QGIS Tool for Shipwreck Detection in Multibeam Sonar Data" presented at OCEANS 2025 Great Lakes. Link to paper: [http://arxiv.org/abs/2509.21386](http://arxiv.org/abs/2509.21386).

For installation instructions, a list of features, and various demos please visit [the wiki page](https://github.com/umfieldrobotics/ShipwreckFinderQGISPlugin/wiki).

*The plugin code is based on 'Create A QGIS Plugin' created by Crabbé Ann, Jakimow Benjamin*
*and Somers Ben and funded by BELSPO STEREO III (Project LUMOS - SR/01/321).*
*The full code is available from https://bitbucket.org/kul-reseco/create-qgis-plugin.*

### SOFTWARE LICENSE

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

### BibTeX

```bibtex
@inproceedings{shep2025shipwreckfinder,
  title={ShipwreckFinder: A QGIS Tool for Shipwreck Detection in Multibeam Sonar Data},
  author={Sheppard, Anja and Smithline, Tyler and Scheffer, Andrew and Smith, David and Sethuraman, Advaith V. and Bird, Ryan and Lin, Sabrina and Skinner, Katherine A.},
  booktitle={Proceedings of OCEANS Great Lakes 2025},
  year={2025},
  organization={IEEE}
}
