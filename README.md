Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches
===================================================================================

The repository contains

- procedures to compute the stereo matching cost with a convolutional neural network;
- procedures to train a convolutional neural network on the stereo matching task;
- a basic stereo method (cross-based cost aggregation, semiglobal matching,
  left-right consistency check, median filter, and bilateral filter); and
- networks that have been trained on the [KITTI
  2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo),
  [KITTI
  2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo),
  and [Middlebury](http://vision.middlebury.edu/stereo/) stereo data sets.

A NVIDIA GPU with at least 6 GB of memory is required to run on the KITTI
data set and 12 GB to run on the Middlebury data set. We tested the code on GTX
Titan (KITTI only), K80, and GTX Titan X.  Note that the network architecture
deviates from the description in the [CVPR
paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Zbontar_Computing_the_Stereo_2015_CVPR_paper.html);
we describe the differences in our upcoming journal paper.  Please cite the
[CVPR
paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Zbontar_Computing_the_Stereo_2015_CVPR_paper.html)
if you use code from this repository in your work.

	@InProceedings{Zbontar_2015_CVPR,
		author = {Zbontar, Jure and LeCun, Yann},
		title = {Computing the Stereo Matching Cost With a Convolutional Neural Network},
		journal = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {June},
		year = {2015}
	}

The code is released under the BSD 2-Clause license.

Computing the Matching Cost
---------------------------

Install [Torch](http://torch.ch/), [OpenCV 2.4](http://opencv.org/), and
[png++](http://www.nongnu.org/pngpp/).

Run all following commands in the same directory as this README file.

Compile the shared libraries:

	$ cp Makefile.proto Makefile
	$ make

The command should produce two files: `libadcensus.so` and `libcv.so`.

To run the stereo algorithm on a stereo pair from the KITTI 2012 training set&mdash;

- Left input image  
  <img src="samples/input/kittiL.png" style="width: 25%;"/>
- Right input image  
  <img src="samples/input/kittiR.png" style="width: 25%;"/>

&mdash;call `main.lua` with the following arguments:

	$ ./main.lua kitti fast -a predict -net_fname net/net_kitti_fast_-a_train_tr.t7 -left samples/input/kittiL.png -right samples/input/kittiR.png -disp_max 70
	Writing right.bin, 1 x 70 x 370 x 1226
	Writing left.bin, 1 x 70 x 370 x 1226
	Writing disp.bin, 1 x 1 x 370 x 1226

The first two arguments (`kitti fast`) are used to set the default
hyperparameters of the stereo method. The outputs are stored as three binary
files:

- `left.bin`: The matching cost after semiglobal matching and cross-based
  cost aggregation with the left image treated as the reference image.  
  <img src="samples/output/left.png" style="width: 25%;"/>
- `right.bin`: Same as `left.bin`, but with the right image treated as the
  reference image.  
  <img src="samples/output/right.png" style="width: 25%;"/>
- `disp.bin`: The disparity map after the full stereo method.  
  <img src="samples/output/disp.png" style="width: 25%;"/>

Use the `bin2png.lua` script to generate the `.png` images like the ones above:

	$ luajit samples/bin2png.lua 
	Writing left.png
	Writing right.png
	Writing disp.png

If you wish to use the raw convolutional neural network outputs, that is,
without applying cross-based cost aggregation and semiglobal matching, run
the following command:

	$ ./main.lua kitti fast -a predict -net_fname net/net_kitti_fast_-a_train_tr.t7 -left samples/input/kittiL.png -right samples/input/kittiR.png -disp_max 70 -sm_terminate cnn
	Writing right.bin, 1 x 70 x 370 x 1226
	Writing left.bin, 1 x 70 x 370 x 1226
	Writing disp.bin, 1 x 1 x 370 x 1226

The resulting disparity maps should look like this:

- `left.png`  
  <img src="samples/output/left_cnn.png" style="width: 25%;"/>
- `right.png`  
  <img src="samples/output/right_cnn.png" style="width: 25%;"/>

### Loading the Output Binary Files ###

You load the binary files by memory mapping them.  We include examples of
memory mapping for some of the more popular programming languages.

- **Lua**

		require 'torch'
		left = torch.FloatTensor(torch.FloatStorage('../left.bin')):view(1, 70, 370, 1226)
		right = torch.FloatTensor(torch.FloatStorage('../right.bin')):view(1, 70, 370, 1226)
		disp = torch.FloatTensor(torch.FloatStorage('../disp.bin')):view(1, 1, 370, 1226)

- **Python**

		import numpy as np
		left = np.memmap('../left.bin', dtype=np.float32, shape=(1, 70, 370, 1226))
		right = np.memmap('../right.bin', dtype=np.float32, shape=(1, 70, 370, 1226))
		disp = np.memmap('../disp.bin', dtype=np.float32, shape=(1, 1, 370, 1226))

- **Matlab**

		left = memmapfile('../left.bin', 'Format', 'single').Data;
		left = reshape(left, [1 70 370 1226]);
		right = memmapfile('../right.bin', 'Format', 'single').Data;
		right = reshape(right, [1 70 370 1226]);
		disp = memmapfile('../disp.bin', 'Format', 'single').Data;
		disp = reshape(right, [1 1 370 1226]);

- **C**

		#include <fcntl.h>
		#include <stdio.h>
		#include <sys/mman.h>
		#include <sys/stat.h>
		#include <sys/types.h>
		int main(void)
		{
			int fd;
			float *left, *right, *disp;
			fd = open("../left.bin", O_RDONLY);
			left = mmap(NULL, 1 * 70 * 370 * 1226 * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
			close(fd);
			fd = open("../right.bin", O_RDONLY);
			right = mmap(NULL, 1 * 70 * 370 * 1226 * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
			close(fd);
			fd = open("../disp.bin", O_RDONLY);
			disp = mmap(NULL, 1 * 1 * 370 * 1226 * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
			close(fd);
			return 0;
		}

Training
--------

This section explains how to train the convolutional neural network on the
KITTI and Middlebury data sets.

### KITTI ###

Download both

- the [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow.zip) data set and unzip it
into `src/data.kitti/unzip` (you should end up with a file `data.kitti/unzip/training/image_0/000000_10.png`) and 
- the [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) data set and unzip it
into `src/data.kitti2015/unzip` (you should end up with a file `data.kitti2015/unzip/training/image_2/000000_10.png`).

Run the preprocessing script:

	$ ./preprocess_kitti.lua

Run `main.lua` to train the network:

	$ ./main.lua kitti2012 fast -action train_tr
	conv(in=1, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	nn.Normalize
	nn.StereoJoin1
	1       0.03037397274128        0.001   3398
	2       0.026550362596851       0.001   6827
	3       0.025564430719268       0.001   10312
	4       0.025006808293515       0.001   13716
	5       0.024617485892804       0.001   17035
	6       0.024294071282911       0.001   20417
	7       0.024049303830253       0.001   23794
	8       0.023861376935358       0.001   27171
	9       0.023688995216277       0.001   30588
	10      0.023537286251383       0.001   33958
	11      0.023441382191954       0.001   37409
	12      0.022877658492854       0.0001  40914
	13      0.022832725849046       0.0001  44352
	14      0.022802679725601       0.0001  47730

The script prints a description of the network and for each of the 14 training
epochs the epoch number, loss on the training set, learning rate, and the time
elapsed in seconds. The network is stored in the `net/` directory.

	$ ls net/
	...
	net_kitti2012_fast_-action_train_tr.t7
	...

### Middlebury ###

Run `download_middlebury.sh` to download the training data
(this can take a long time, depending on your internet connection).

	$ ./download_middlebury.sh

Compile the [MiddEval3-SDK](http://vision.middlebury.edu/stereo/submit3/). You
should end up with the `computemask` binary in one of the directories listed in
your `PATH` enviromential variable.  Run the preprocessing script:

	$ ./preprocess_mb.py imperfect gray

Use `main.lua` to train the network:

	$ ./main.lua mb slow -a train_tr

Other Useful Commands
---------------------

Compute the loss on the validation set:

	$ ./main.lua kitti fast -a test_te -net_fname net/net_kitti_fast_-a_train_tr.t7 
	kitti fast -a test_te -net_fname net/net_kitti_fast_-a_train_tr.t7 
	0.86836290359497        0.0082842716717202
	...
	0.73244595527649        0.024202708004929
	0.72730183601379        0.023603160822285
	0.030291934952454

The validation error rate of the fast architecture on the KITTI 2012 data set is 3.02%.

\***

Prepare files for submission to the KITTI and Middlebury evaluation server.

	$ ./main.lua kitti fast -a submit -net_fname net/net_kitti_fast_-a_train_tr.t7 
	kitti fast -a submit -net_fname net/net_kitti_fast_-a_train_tr.t7 
	  adding: 000038_10.png (deflated 0%)
	  adding: 000124_10.png (deflated 0%)
	  ...
	  adding: 000021_10.png (deflated 0%)

The output is stored in `out/submission.zip` and can be used to submit to the
[KITTI evaluation
server](http://www.cvlibs.net/datasets/kitti/user_submit.php).

\***

Experiment with different network architectures:

	$ ./main.lua kitti slow -a train_tr -l1 2 -fm 128 -l2 3 -nh2 512
	kitti slow -a train_tr -l1 2 -fm 128 -l2 3 -nh2 512 
	conv(in=1, out=128, k=3)
	cudnn.ReLU
	conv(in=128, out=128, k=3)
	cudnn.ReLU
	nn.Reshape(128x256)
	nn.Linear(256 -> 512)
	cudnn.ReLU
	nn.Linear(512 -> 512)
	cudnn.ReLU
	nn.Linear(512 -> 512)
	cudnn.ReLU
	nn.Linear(512 -> 1)
	cudnn.Sigmoid
	...

\***

Measure the runtime on a particular data set:

	$ ./main.lua kitti fast -a time
	kitti fast -a time 
	conv(in=1, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	nn.Normalize2
	nn.StereoJoin1
	0.73469495773315

It take 0.73 seconds to run the fast architecure on the KITTI 2012 data set. If
you care only about the time spent in the neural network, you can terminate the
stereo method early:

	$ ./main.lua kitti fast -a time -sm_terminate cnn
	kitti fast -a time -sm_terminate cnn 
	conv(in=1, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	cudnn.ReLU
	conv(in=64, out=64, k=3)
	nn.Normalize2
	nn.StereoJoin1
	0.31126594543457
