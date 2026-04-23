from isaacsim import SimulationApp
# Boot up the GUI
app = SimulationApp({"headless": False})

# Keep the window open until you close it manually
while app.is_running():
    app.update()

app.close()