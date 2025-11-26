# Manga Colorizer (Standalone GUI Fork)

This project provides a standalone desktop application with a graphical user interface (GUI) for the [Manga-Colorizer](https://github.com/BinitDOX/Manga-Colorizer). It functions as a UI wrapper, removing the original client/server architecture that required running a backend server and a browser-based front end (like Firefox, Safari, or Chrome).

![App Screenshot](Screenshot/ss1.png)
![Preview Screenshot](Screenshot/ss2.png)

Unlike the original version which colorized images directly on websites, this application allows you to process local image files from any folder on your computer. This provides a streamlined, offline-first workflow for colorizing, denoising, and upscaling your manga pages, allowing you to process entire folders of images with just a few clicks.

**TL;DR**
- **Original:** Runs a server + uses your web browser as a plugin to process pages.  
- **This fork:** Runs as a desktop app — no browser, no separate server.  
- **Original:** Processes web pages individually.  
- **This fork:** Can process entire folders at once.  

---

## 📦 Full Installation Guide

Follow these steps to set up and run the application on your local machine.

### 1️⃣ Clone the Repository

Clone this repository with Git:

```bash
git clone https://github.com/sepTN/Manga-Colorizer-GUI
cd Manga-Colorizer-GUI
```

Or download as a ZIP and extract it.

---

### 2️⃣ Set Up the Conda Environment (optional)

We recommend **Conda** to manage dependencies.

**Create a new environment:**
```bash
conda create --name manga-colorizer python=3.9
```

**Activate it:**
```bash
conda activate manga-colorizer
```

---

### 3️⃣ Install Required Libraries

**Standard Packages:**
```bash
pip install Flask Flask_Cors matplotlib numpy opencv_python_headless scikit_image einops
```

**PyTorch Installation:** choose the correct one for your system.

- **Mac (Apple Silicon M1/M2/M4):**
  ```bash
  pip install torch torchvision
  ```
- **Windows/Linux with NVIDIA GPU (CUDA):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **Windows/Linux without a dedicated GPU:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

---

### 4️⃣ Download the AI Models

Download and place this `generator.zip` file in `Backend/networks/`.

- **Colorizer Model:**  
  [Generator weights](https://drive.google.com/file/d/1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t/view?usp=sharing) → `generator.zip`

- **Upscaler Models:**
  - ESRGAN (default, already included): `RealESRGAN_x4plus_anime_6B.pt`
  - GigaGAN (optional): I haven’t tested this model myself, so you’re on your own here — in the app’s Advanced Settings, you can set the path to the model file manually.

---

### 5️⃣ Run the Application

From the main project folder:
```bash
python app.py
```

The GUI will appear—start colorizing your manga! 🎨

---

## 🙌 Credits

Built on the work of amazing developers:

- [BinitDOX](https://github.com/BinitDOX) – Original [Manga-Colorizer](https://github.com/BinitDOX/Manga-Colorizer)
- [qweasdd](https://github.com/qweasdd) – [manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2)
- [xiaogdgenuine](https://github.com/xiaogdgenuine) – [Manga-Colorization-FJ](https://github.com/xiaogdgenuine/Manga-Colorization-FJ)
- [xinntao](https://github.com/xinntao) – [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [vatavian](https://github.com/vatavian) – Improved fork with on-the-fly colorization
- [iG8R](https://github.com/iG8R) – Testing and feedback
- Forked and streamlined by [sepTN](https://github.com/sepTN) for a simpler all-in-one experience.
