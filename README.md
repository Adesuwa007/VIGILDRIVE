# VIGILDRIVE  
### Real-Time Driver Drowsiness Detection System

VIGILDRIVE is a real-time system that checks driver attention and drowsiness using computer vision. It monitors eye behavior through a webcam and alerts the driver when it spots signs of fatigue. The system is lightweight and can run on any standard machine without special hardware. Driver drowsiness is a major cause of road accidents worldwide. It often results from fatigue, long hours of driving, or nighttime driving. Many current solutions are expensive, vehicle-specific, or hard to get. VIGILDRIVE solves this problem by offering an affordable camera-based solution that works in real time.

The system captures live video from a webcam, detects facial features using MediaPipe Face Mesh, calculates the Eye Aspect Ratio (EAR) in real time, and identifies prolonged eye closure. It triggers visual and audio alerts when it detects drowsiness. It also automatically turns on night vision in low-light situations to maintain accurate detection. VIGILDRIVE includes a webcam monitoring, EAR-based detection, instant audio alerts, an automatic night vision mode, a modern web interface created with Gradio, lightweight processing that is easy on CPU use, and it is fully open-source.

The project uses Python, OpenCV, MediaPipe, Gradio, NumPy, SciPy, and Pygame.

To install and run VIGILDRIVE, first clone the repository with this command:  
> git clone https://github.com/Adesuwa007/VIGILDRIVE.git  
> cd VIGILDRIVE  

It is a good idea to create a virtual environment:  
> python -m venv venv  

Next, activate the virtual environment.  
On Windows:  
> venv\Scripts\activate  
On macOS or Linux:  
> source venv/bin/activate  

Then, upgrade pip and install the necessary dependencies:  
> python -m pip install --upgrade pip  
> python -m pip install -r requirements.txt  

If needed, manually install or upgrade Gradio and Pygame:  
> python -m pip install --upgrade gradio  
> pip install pygame  

To run the application, use this command:  
> python driver_drowsiness_web.py  

After starting the application, open your browser and go to:  
> http://127.0.0.1:7860  

The system requires Python 3.10 or later, a working webcam, and a compatible operating system like Windows, macOS, or Linux. VIGILDRIVE is suitable for long-distance drivers, improving night driving safety, truck and taxi drivers, driver training systems, and driving simulators. Its uniqueness lies in the absence of special hardware. It works solely with a standard webcam, provides real-time alerts, improves night vision, and allows for easy extension. This project is licensed under the MIT License.

VIGILDRIVE was created for hackathon submissions, Contributions are welcome. Feel free to fork the repository and submit pull requests. If you find this project useful, please consider starring the repository.

VIGILDRIVE - Drive Safe. Stay Awake.
