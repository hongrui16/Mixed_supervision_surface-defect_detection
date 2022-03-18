## Acknowledgement
This is a modified versiono of [mixed-segdec-net-comind2021](https://github.com/vicoslab/mixed-segdec-net-comind2021.git).

## Main new feature
A commonly used dataset dataloader is added, as well as other useful functions.
You can parpare your dataset format as following:
```
/datasets
    /train
        /image
            a.jpg
            ...
        /label
            a.png
            ...
    /val
        /image
            b.jpg
            ...
        /label
            b.png
            ...
    /test
        /image
            c.jpg
            ...
        /label
            c.png
            ...
            
```
Specifiy the dataset dir, or you can give image filepath in a txt file as following:

```
train.txt
    train_dir/a.jpg
    ...

val.txt
    val_dir/b.jpg
    ...

test.txt
    test_dir/c.jpg
    ...   
            
```
then, run trainPot.sh to start training.

For the origin metal surface abnormal detection, you can follow the indroduction in [mixed-segdec-net-comind2021](https://github.com/vicoslab/mixed-segdec-net-comind2021.git).