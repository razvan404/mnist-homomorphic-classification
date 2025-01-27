import threading

import requests
import pygame
import numpy as np

from src.ckks import Encryptor


class DemoClient:
    WINDOW_SIZE = 560
    GRID_SIZE = 28  # For a 1x1 grid, scaled up to 28x28 pixels for MNIST-like inference
    PIXEL_SIZE = WINDOW_SIZE // GRID_SIZE
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LIGHT_GRAY = (200, 200, 200)
    BORDER_COLOR = (0, 0, 0)
    PREDICT_BUTTON_COLOR = (100, 100, 255)
    RESET_BUTTON_COLOR = (255, 100, 100)
    TEXT_COLOR = (255, 255, 255)
    FONT_SIZE = 28
    URL = "http://localhost:5000/inference"

    def __init__(self):
        pygame.init()

        # Initialize the screen
        self.screen = pygame.display.set_mode(
            (self.WINDOW_SIZE, self.WINDOW_SIZE + 150)
        )
        pygame.display.set_caption("Draw a Number")
        self.font = pygame.font.Font(None, self.FONT_SIZE)

        # Button dimensions
        button_width = self.WINDOW_SIZE // 2.5
        button_left = self.WINDOW_SIZE // 20
        button_right = self.WINDOW_SIZE - button_left - button_width

        self.predict_button_rect = pygame.Rect(
            button_left, self.WINDOW_SIZE + 10, button_width, 30
        )
        self.reset_button_rect = pygame.Rect(
            button_right, self.WINDOW_SIZE + 10, button_width, 30
        )

        # Canvas to draw the digit
        self.drawing = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)

        # Placeholder for predictions
        self.probabilities = [0.0] * 10
        self.prediction_text = "Prediction: None"

        # CKKS Encryptor
        self.encryptor = Encryptor()
        self.serialized_encryptor = self.encryptor.serialize()

    def draw_grid(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = self.BLACK if self.drawing[y, x] == 0 else self.WHITE
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x * self.PIXEL_SIZE,
                        y * self.PIXEL_SIZE,
                        self.PIXEL_SIZE,
                        self.PIXEL_SIZE,
                    ),
                )
        # Draw border around the drawing area
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            (0, 0, self.WINDOW_SIZE, self.WINDOW_SIZE),
            5,
        )

    def draw_buttons(self):
        # Draw prediction button
        pygame.draw.rect(
            self.screen, self.PREDICT_BUTTON_COLOR, self.predict_button_rect
        )
        predict_text_surface = self.font.render("Predict", True, self.TEXT_COLOR)
        predict_text_rect = predict_text_surface.get_rect(
            center=self.predict_button_rect.center
        )
        self.screen.blit(predict_text_surface, predict_text_rect)

        # Draw reset button
        pygame.draw.rect(self.screen, self.RESET_BUTTON_COLOR, self.reset_button_rect)
        reset_text_surface = self.font.render("Reset", True, self.TEXT_COLOR)
        reset_text_rect = reset_text_surface.get_rect(
            center=self.reset_button_rect.center
        )
        self.screen.blit(reset_text_surface, reset_text_rect)

    def mouse_in_canvas(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        return 0 < mouse_x < self.WINDOW_SIZE and 0 < mouse_y < self.WINDOW_SIZE

    def iterate_pencil_coords(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()

        grid_x = (mouse_x - self.PIXEL_SIZE // 2) // self.PIXEL_SIZE
        grid_y = (mouse_y - self.PIXEL_SIZE // 2) // self.PIXEL_SIZE

        for dx in range(2):
            for dy in range(2):
                if (
                    0 <= grid_x + dx < self.GRID_SIZE
                    and 0 <= grid_y + dy < self.GRID_SIZE
                ):
                    yield grid_x + dx, grid_y + dy

    def draw_preview(self):
        for x, y in self.iterate_pencil_coords():
            pygame.draw.rect(
                self.screen,
                self.LIGHT_GRAY,
                (
                    x * self.PIXEL_SIZE,
                    y * self.PIXEL_SIZE,
                    self.PIXEL_SIZE,
                    self.PIXEL_SIZE,
                ),
            )

    def draw_probabilities(self):
        start_x = 30
        start_y = self.WINDOW_SIZE + 80
        radius = 25
        spacing = 55

        for i, prob in enumerate(self.probabilities):
            gray_value = int((1 - prob) * 255)
            color = (gray_value, gray_value, gray_value)

            pygame.draw.circle(
                self.screen, color, (start_x + i * spacing, start_y), radius
            )
            pygame.draw.circle(
                self.screen,
                self.BORDER_COLOR,
                (start_x + i * spacing, start_y),
                radius,
                2,
            )

            digit_surface = self.font.render(str(i), True, self.BLACK)
            digit_rect = digit_surface.get_rect(center=(start_x + i * spacing, start_y))
            self.screen.blit(digit_surface, digit_rect)

        # Draw prediction text
        pred_surface = self.font.render(self.prediction_text, True, self.BLACK)
        pred_rect = pred_surface.get_rect(
            center=(self.WINDOW_SIZE // 2, self.WINDOW_SIZE + 125)
        )
        self.screen.blit(pred_surface, pred_rect)

    def set_prediction_text(self, text: str):
        self.prediction_text = f"Prediction: {text}"

    def process_inference_response(self, response: requests.Response):
        serialized_encrypted_preds = response.json().get("preds", None)
        if serialized_encrypted_preds is None:
            self.probabilities = [0.0] * 10
            self.set_prediction_text("None")
            return
        encrypted_preds = self.encryptor.deserialize_data(serialized_encrypted_preds)
        preds = self.encryptor.decrypt(encrypted_preds)

        # Perform softmax
        e_preds = np.exp(preds - np.max(preds))
        self.probabilities = (e_preds / np.sum(e_preds)).tolist()

        # Determine predicted digit and update prediction text
        predicted_digit = np.argmax(self.probabilities)
        confidence = self.probabilities[predicted_digit] * 100
        self.set_prediction_text(f"{predicted_digit} ({confidence:.2f}%)")

    def send_inference_request(self):
        try:
            encrypted_image = self.encryptor.encrypt_image(self.drawing)
            serialized_encrypted_image = self.encryptor.serialize_data(encrypted_image)

            request_body = {
                "encryptor": self.serialized_encryptor,
                "image": serialized_encrypted_image,
            }

            response = requests.post(self.URL, json=request_body)

            if response.status_code == 200:
                self.process_inference_response(response)
            else:
                print(f"Invalid response status: {response.status_code}")
                self.set_prediction_text("Error")
        except Exception as e:
            print(f"Error: {e}")
            self.set_prediction_text("Error")

    def run(self):
        running = True
        while running:
            self.screen.fill(self.WHITE)

            self.draw_grid()
            self.draw_buttons()
            self.draw_probabilities()

            if self.mouse_in_canvas():
                self.draw_preview()

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    or event.type == pygame.MOUSEMOTION
                    and pygame.mouse.get_pressed()[0]
                ):
                    if self.mouse_in_canvas():
                        for x, y in self.iterate_pencil_coords():
                            self.drawing[y, x] = 1.0

                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if self.predict_button_rect.collidepoint(event.pos):
                            self.set_prediction_text("Loading...")
                            threading.Thread(target=self.send_inference_request).start()
                        elif self.reset_button_rect.collidepoint(event.pos):
                            self.drawing.fill(0.0)
                            self.probabilities = [0.0] * 10
                            self.set_prediction_text("None")

        pygame.quit()


if __name__ == "__main__":
    app = DemoClient()
    app.run()
