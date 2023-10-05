# :::: iid, no compression ::::::::
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --frac 0.05
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --frac 0.05
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --frac 0.05
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --frac 0.05
# ::python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --frac 0.05

# use this for dynamic concept
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --frac 0.2 --num_users 10 --concept dynamic

python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --frac 0.2 --num_users 10
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --frac 0.2 --num_users 10
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --frac 0.2 --num_users 10
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --frac 0.2 --num_users 10
# :::: non-iid, no compression :::::::::
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --num_users 10
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --num_users 10
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --frac 0.2
# :::: iid, light compression ::::::::
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 3
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 3
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 3
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 3
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 3  
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 3 --frac 0.2
# :::: non-iid, light compression :::::::::
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 3
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 3
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 3
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 3
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 3
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 3 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 3 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 3 --frac 0.2
# :::: iid, Heavy compression ::::::::
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 1
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 1
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 1
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 1
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 1  
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 1 --frac 0.2
# :::: non-iid, Heavy compression :::::::::
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 1
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 1
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 1
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 1
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 1
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 1 --frac 0.05
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched RS --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BC --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched BN2 --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1 --comp Spar -- blocks 1 --frac 0.2
python main_fed.py --dataset cifar --num_channels 1 --model cnn --epochs 30 --gpu 0  --sched G1-M --comp Spar -- blocks 1 --frac 0.2