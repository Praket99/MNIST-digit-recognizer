import tkinter as tk
import numpy as np

from neural_network import NeuralNetwork

# SETTINGS

SIZE = 28
SPACING = 21
PIXEL_SIZE = 14
BRUSH_RADIUS = 1.5

# LOAD TRAINED MODEL

network = NeuralNetwork()

model = np.load("trained_model.npz")

network.W1 = model["W1"]
network.b1 = model["b1"]

network.W2 = model["W2"]
network.b2 = model["b2"]

network.W3 = model["W3"]
network.b3 = model["b3"]

print("Trained model loaded!")

# PIXEL DATA

pixels = [
    [0.0 for _ in range(SIZE)]
    for _ in range(SIZE)
]

# WINDOW

root = tk.Tk()

root.title("My Neural Network - Digit Recognizer")

canvas_size = SIZE * SPACING

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black"
)

canvas.pack(padx=10, pady=10)

# CREATE GRID

squares = [
    [None for _ in range(SIZE)]
    for _ in range(SIZE)
]


def grayscale(value):

    value = int(value * 255)

    return f"#{value:02x}{value:02x}{value:02x}"


for y in range(SIZE):

    for x in range(SIZE):

        x1 = x * SPACING
        y1 = y * SPACING

        x2 = x1 + PIXEL_SIZE
        y2 = y1 + PIXEL_SIZE

        squares[y][x] = canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill="#000000",
            outline=""
        )


# DRAW

def draw(event):

    grid_x = event.x // SPACING
    grid_y = event.y // SPACING

    for y in range(
        max(0, int(grid_y - BRUSH_RADIUS)),
        min(SIZE, int(grid_y + BRUSH_RADIUS + 1))
    ):

        for x in range(
            max(0, int(grid_x - BRUSH_RADIUS)),
            min(SIZE, int(grid_x + BRUSH_RADIUS + 1))
        ):

            dx = x - grid_x
            dy = y - grid_y

            distance = np.sqrt(dx * dx + dy * dy)

            if distance <= BRUSH_RADIUS:

                intensity = np.exp(
                    -(distance ** 2) / 2
                )

                pixels[y][x] = max(
                    pixels[y][x],
                    intensity
                )

                canvas.itemconfig(
                    squares[y][x],
                    fill=grayscale(pixels[y][x])
                )


canvas.bind("<B1-Motion>", draw)
canvas.bind("<Button-1>", draw)

# CLEAR

def clear():

    for y in range(SIZE):

        for x in range(SIZE):

            pixels[y][x] = 0.0

            canvas.itemconfig(
                squares[y][x],
                fill="#000000"
            )

    prediction_label.config(
        text="Prediction: -"
    )

    confidence_label.config(
        text="Confidence: -"
    )

def preprocess_image():

    # Convert pixels to NumPy array
    image = np.array(
        pixels,
        dtype=np.float32
    )

    # Find pixels that actually belong to the digit
    coords = np.argwhere(image > 0.05)

    # Nothing drawn
    if len(coords) == 0:
        return image.reshape(784)

    # Bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the digit
    cropped = image[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Create a new 28 × 28 image
    result = np.zeros((28, 28), dtype=np.float32)

    # Find scale so digit fits nicely
    height, width = cropped.shape

    scale = 20 / max(height, width)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    # Resize using PIL
    from PIL import Image

    cropped_image = Image.fromarray(
        (cropped * 255).astype(np.uint8)
    )

    resized = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    resized = np.array(
        resized,
        dtype=np.float32
    ) / 255.0

    # Center the digit
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    result[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return result.reshape(784)

# PREDICT

def predict():

    # Preprocess drawing to look more like MNIST
    image = preprocess_image()

    # Debug

    print("\nPixel range:")
    print("Minimum:", image.min())
    print("Maximum:", image.max())
    print("Non-zero pixels:", np.count_nonzero(image))

    # Neural network

    probabilities = network.forward(image)

    prediction = np.argmax(probabilities)

    confidence = probabilities[prediction] * 100

    # Result

    prediction_label.config(
        text=f"Prediction: {prediction}"
    )

    confidence_label.config(
        text=f"Confidence: {confidence:.2f}%"
    )

    # Show all probabilities

    print("\nPrediction:", prediction)
    print("Confidence:", f"{confidence:.2f}%")

    print("\nProbabilities:")

    for digit in range(10):

        print(
            f"{digit}: "
            f"{probabilities[digit] * 100:.2f}%"
        )

# BUTTONS

button_frame = tk.Frame(root)

button_frame.pack(pady=10)


predict_button = tk.Button(
    button_frame,
    text="PREDICT",
    command=predict,
    width=15
)

predict_button.grid(
    row=0,
    column=0,
    padx=5
)


clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    command=clear,
    width=15
)

clear_button.grid(
    row=0,
    column=1,
    padx=5
)

# RESULT

prediction_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Arial", 24)
)

prediction_label.pack()


confidence_label = tk.Label(
    root,
    text="Confidence: -",
    font=("Arial", 14)
)

confidence_label.pack(pady=(0, 15))


# START

root.mainloop()