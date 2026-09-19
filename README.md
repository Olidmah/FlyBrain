# FlyBrain

A Geometry Dash AI project powered by FlyBrain.


# 🧠 FlyBrain

FlyBrain is a Geometry Dash AI project that learns to play using screen vision and a simulated fruit-fly brain.

## 🎮 Easy Installation — Windows

The easiest way to use FlyBrain is the Windows EXE.

1. Open the **Releases** section of this repository.
2. Download **FlyBrain.exe**.
3. Put it somewhere on your computer.
4. Double-click **FlyBrain.exe**.
5. Log in or create a FlyBrain account.
6. Start Geometry Dash.
7. Choose the FlyBrain mode you want.

## 🛑 Emergency Stop

Press:

**F8**

This immediately stops FlyBrain's automated input.

## 🐍 Developer Installation

You need Python installed on Windows.

Check your Python installation with:

```powershell
py --version
```

Then download this repository and open PowerShell in the project folder.

Install the required packages:

```powershell
py -m pip install -r requirements.txt
```

Run FlyBrain:

```powershell
py app\flybrain_app.py
```

## 📦 Build the Windows EXE

Install PyInstaller:

```powershell
py -m pip install pyinstaller
```

Build FlyBrain:

```powershell
py -m PyInstaller --onefile --windowed --name FlyBrain app\flybrain_app.py
```

The finished program will appear in:

```text
dist\FlyBrain.exe
```

## 🎮 Modes

### Observe

FlyBrain watches Geometry Dash without controlling it.

### Learn

FlyBrain collects training information and learns from gameplay.

### AI Test

FlyBrain makes decisions but does not control the game.

### AI Play

FlyBrain controls Geometry Dash.

### Entertainment Mode

FlyBrain runs in a small window in the bottom-right while it plays Geometry Dash.

## 🧠 Simple vs Complex AI

**Simple AI**

* Uses simpler decision-making.
* Faster and easier to test.

**Complex AI**

* Uses the trained model and FlyBrain processing.
* Intended for more advanced experiments.

## 🔗 Account Linking

FlyBrain can have optional account-linking features.

Never enter your Geometry Dash password into FlyBrain.

## 🔒 Personal Files

The following files are personal and should **not** be uploaded to GitHub:

```text
flybrain_accounts.json
flybrain_learning.json
```

These contain local account and learning information.

## 🛠️ Troubleshooting

### FlyBrain won't start

Try:

```powershell
py -m pip install -r requirements.txt
```

Then run:

```powershell
py app\flybrain_app.py
```

### EXE won't start

Try running the Python version from PowerShell so an error message can be seen:

```powershell
py app\flybrain_app.py
```

### AI isn't controlling Geometry Dash

Check that:

* Geometry Dash is running.
* FlyBrain is running.
* The correct AI mode is selected.
* Geometry Dash is visible to FlyBrain.
* F8 has not been triggered.

## 👨‍💻 Development

This repository is private.

Only invited collaborators can access the source code.

## ⚠️ Safety

FlyBrain can automatically control keyboard and mouse input.

Keep the F8 emergency stop available and do not leave automated control running when it could interfere with other programs.

## 📜 Project

FlyBrain is an experimental project for computer vision, machine learning, and simulated-brain research.
