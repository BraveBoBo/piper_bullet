# 使用 PyBullet 实现机械臂抓取

## 介绍

本项目演示了如何使用PyBullet库实现一个机械臂抓取任务。PyBullet是一个开源的物理仿真工具，支持刚体物理仿真、碰撞检测、机器人控制等功能。我们将使用PyBullet从头开始构建一个机械臂模型，并在仿真环境中执行抓取操作。

## 项目结构
robot_grasping_project ├── README.md # 项目说明文件 ├── grasping_script.py # 主要抓取实现脚本 ├── robot_model.urdf # 机械臂URDF模型文件 ├── environment.py # 设置仿真环境的脚本 └── requirements.txt # 项目依赖文件

## 安装依赖

### 1. 创建虚拟环境

首先，建议为本项目创建一个虚拟环境来管理依赖。你可以通过以下命令创建和激活虚拟环境：

```bash
conda activate bullet
```

### 2. build 


