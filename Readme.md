# Accelerating Outlier-robust Rotation Estimation by Stereographic Projection

## 📝 **About**

This paper is concerned with robustly solving one as well as multiple rotation estimates in the presence of a large number of outliers and noise.

<div align=center>

![我的头像](https://i.imgur.com/cJiYF7P.png)

<div align=left>

The rotation estimation problem is decoupled into two subproblems, i.e., rotation axis and angle, by investigating a special geometric constraint. In other words, solving the rotation estimation problem with three degrees of freedom is transformed into solving the rotation axis with two degrees of freedom and the rotation angle with one degree of freedom. Consequently, the efficiency of solving the rotation estimation problem is significantly improved.

<div align=center>

![我的头像](https://i.imgur.com/YYDDe0c.png)

<div align=left>

By considering the rotation geometric constraint, solving the rotation axis becomes finding the points of maximum intersection of circles on the unit sphere. Innovatively, we use stereographic projection to map the circles from a three-dimensional sphere onto a two-dimensional plane. In this way, computations in redundant spaces can be avoided, therefore increasing the efficiency of the solution.

<div align=center>
  
![我的头像](https://i.imgur.com/saoqhjL.png)

<div align=left>
To robustly and efficiently solve the rotation axes, we introduce a spatial voting strategy to find points of maximum intersection of circles on the 2D plane. Using this strategy, we can find the optimal rotation axis and multiple rotation axes simultaneously.

---

## 📝 **Algorithm**

<div align=center>

![我的头像](https://i.imgur.com/U58WCma.png)

<div align=left>

---

## 📝 **Comparative experiments on robust rotation estimation**

### 🍀 **Synthetic data experiments**

To evaluate the accuracy and robustness of the proposed method, comparative experiments are systematically conducted using controlled synthetic and real-world data. All experiments are validated on a PC with 16G RAM and GeForce RTX 4080. Unless otherwise specified, each synthetic experiment was conducted 200 times.

<div align=center>

![我的头像](https://i.imgur.com/eoWyKzM.png)

<div align=left>

<div align=center>

![我的头像](https://i.imgur.com/XukCbei.png)

<div align=left>

### 🍀 **Real_world data experiments**

We use two distinct datasets to demonstrate our algorithm's performance in both indoor and outdoor environments: 3DMatch and KITTI.

<div align=center>

![我的头像](https://i.imgur.com/pDrOocJ.png)

<div align=left>

<div align=center>

![我的头像](https://i.imgur.com/EEybojD.png)

<div align=left>
<div align=center>

![我的头像](https://i.imgur.com/xemSAjZ.png)

<div align=left>

---

## 📝 **Comparative experiments on multiple rotations**

### 🍀 **Synthetic data experiments**

<div align=center>

![我的头像](https://i.imgur.com/kmnY54w.png)

<div align=left>

### 🍀 **Real_world data experiments**

<div align=center>

![我的头像](https://i.imgur.com/8IVI5wO.png)

<div align=left>
<div align=center>
  
![我的头像](https://i.imgur.com/2Y5JFfm.png)

<div align=left>

---

## 🌟 **Install**

This program uses python 3.11. Make sure that your Python environment is activated (e.g. with virtualenv or conda)

`pip install cupy`
`pip install numpy`
`pip install scipy`

**For CuPy, make sure that CUDA is installed correctly and is compatible with the CuPy version.**

---

## 🏁 **Running**

**Copy the main.py file to pycharm or vscode to run the result directly.**

## 🏁 **Contact me**

If you have any questions about this code or paper, feel free to contact me at
[mc35344atconnect.um.edu.mo](mc35344@connect.um.edu.mo).



