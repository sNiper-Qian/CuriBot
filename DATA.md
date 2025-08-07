ShapeNetCore data release v2 (Fall 2016)

This second release of ShapeNetCore is an update to ShapeNetCore v1 with improved quality of model geometry and fixed issues relating to materials and textures. The files contain ShapeNetCore: a densely annotated subset of ShapeNet released to the research community. Each zip file is named by the synset noun offset in WordNet (version 3.0) as an eight-digit zero padded string.  For example, bench is contained within 02828884.zip since the WordNet synset offset for bench is 02828884 (you can browse WordNet at http://wordnetweb.princeton.edu/perl/webwn3.0).  The corresponding ImageNet synsets can be accessed at http://www.image-net.org/synset?wnid=n<synsetId> where <synsetId> is replaced by the synset offset (note that ImageNet includes an 'n' prefix for noun synsets). 

Within each synset zip file there is a set of OBJ files of all the 3D models annotated under that synset.  Each model is under a directory named after its source id, which is the id of the original model on the online repository from which it was collected.  Within the model directory, you will find texture images, json,mtl,obj,solid.binvox and binvox models and png screenshots files.  You can view OBJ mesh files with software such as Assimp, the Open Asset Import Library (http://assimp.sourceforge.net/). The v2 package provides improved .obj+mtl format model data that replaces the .obj+.mtl data in ShapeNetCore v1. 

NOTE: The OBJ files are now pre-aligned so that the up direction is the +Y axis, and the front is the -Z axis (this fixes the mirroring issue in ShapeNetCore v1). The Y-Z plane is the bilateral symmetry plane for most categories.

The taxonomy.json file in ShapeNetCore v2 contains a simple JSON format representation of the ShapeNetCore synset taxonomy indicating for each synset the synset offset (synsetId), the synset lemma (name), an array of the children synsets ids (children), and the total number of model instances (numInstances).  This JSON file is obtained from the more comprehensive ShapeNet taxonomy JSON at https://www.shapenet.org/resources/data/shapenetcore.taxonomy.json by filtering with the command:

```
jq "[.[] | recurse (.children[]?) | \{synsetId: .metadata.name, name: .metadata.label, children: [.children[]?.metadata.name], numInstances: .metadata.numInstances \}]" shapenetcore.taxonomy.json
```

(requires the JSON filter library `jq` which can be obtained from http://stedolan.github.io/jq/ , a linux x64 binary is included in this package)


Last updated: 2018-05-04

v0 (Fall 2016)
- Initial release