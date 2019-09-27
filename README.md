# SQL Dataset for Pytorch
An SQL databse can not easily be used in an Dataset for Pytorch. If converting the database to another format is not desired, this code implement two ways to use the database (sqllite) directly, without loading the hole database into RAM.
The first option loads each observation idividualy, whcih is a huge performance bottleneck. The second preloads a part of the database into RAM, this significantly increases performance.

