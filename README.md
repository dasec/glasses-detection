# glasses-detection
This repository contains code for automatic detection of glasses in near-infrared images implemented by Florian Struck.

## License
This work is licensed under license provided by Hochschule Darmstadt ([h_da-License](/hda-license.pdf)).

## Attribution
Any publications using the code must cite and reference the conference paper [1].

## Contents
This repository contains 3 different approaches for glasses detection:
<ul>
	<li>explicit-glasses-identifier - an explicit approach for glasses detection based on edges and reflections</li>
	<li>dl-glasses-identifier - uses deep neuronal network to identify glasses</li>
	<li>statistic-glasses-identifier - uses the BSIF filter and statistical metrics of an image to identify glasses</li>
</ul>

## Instructions
The repository contains 3 independent projects.
Each project has its own structure and dependencies and can therefore be built independently of the other projects.
They can be built by running the "make" command in their respective project folders.
Afterwards, the executable can be found in PROJECT/build/.

## Dependencies
explicit-glasses-identifier:
<ul>
	<li>BOOST library (Version >= 1.52)</li>
	<li>OpenCV (Version 2.4)</li>
	<li>glog (Version 0.3.5)</li>
</ul>

dl-glasses-identifier:
<ul>
	<li>BOOST library (Version >= 1.52)</li>
	<li>OpenCV (Version 2.4)</li>
	<li>glog (Version 0.3.5)</li>
	<li>Caffe (See <a href="http://caffe.berkeleyvision.org/installation.html">http://caffe.berkeleyvision.org/installation.html</a>)</li>
</ul>

statistic-glasses-identifier:
<ul>
	<li>BOOST library (Version >= 1.52)</li>
	<li>OpenCV (Version 2.4)</li>
	<li>glog (Version 0.3.5)</li>
	<li>matio (Version 1.5)</li>
</ul>

## Contact
Code author: Florian Struck (florian.struck@stud.h-da.de)

## References
* [1] Pawel Drozdowski, Florian Struck, Christian Rathgeb, Christoph Busch: "Detection of Glasses in Near-infrared Ocular Images", in Proc. of the 11th IAPR International Conference on Biometrics (ICB 2018), Queensland, Australia, February 2018.

© [Hochschule Darmstadt](https://www.h-da.de/ "Hochschule Darmstadt website")
