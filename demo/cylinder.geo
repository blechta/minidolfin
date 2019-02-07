SetFactory("OpenCASCADE");

Mesh.Algorithm = 6;
Mesh.CharacteristicLengthMin = 0.15;
Mesh.CharacteristicLengthMax = 0.15;

Cylinder(1) = {-10,0,0, 10,0,0, 1.0};
Cylinder(2) = {-10,0,0, 10,0,0, 0.3};

BooleanDifference(3) = {Volume{1}; Delete;}{Volume{2};};

Coherence;

Physical Volume(3) = {3};
Physical Volume(2) = {2};
