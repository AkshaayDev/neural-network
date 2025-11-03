/*
This is a simple 28x28 grid paint app that shows the network's guess of the digit drawn
This script reads the network from a file `nn.dat` and uses it to predict the digit
This script also requires a font file (`arial.ttf` in this case) to display the text

Usage:
1. Download SFML from the official website: https://www.sfml-dev.org/download/
2. Follow the instructions to set up an SFML project
3. Copy and paste this code into the main C++ file
4. Ensure that a `nn.dat` file exists in the same directory
5. Add a font file of a type supported by SFML and include its path in `fontPath`
6. Compile and run

Controls:
Left Mouse: White brush
Right Mouse: Black brush/Erase
'C' Key: Clear canvas
*/
#include <SFML/Graphics.hpp>
#include "../../neural-network.hpp"
#include <iostream>
#include <string>
#include <fstream>

const std::string fontPath = "./arial.ttf";
const int squareSize = 20; // Display size of a pixel on the grid
const int brushSize = 2;
// Represents a flattened column matrix of the grid
NNMatrix img(28*28,1);

static void draw(sf::RenderWindow& window, int white) {
	int x = sf::Mouse::getPosition(window).x / squareSize;
	int y = sf::Mouse::getPosition(window).y / squareSize;
	if (x < 0 || x >= 28 || y < 0 || y >= 28) return;

	for (int i = -brushSize; i < brushSize; i++) {
		for (int j = -brushSize; j < brushSize; j++) {
			int nx = x + i, ny = y + j;
			// Make sure current pixel is in the grid
			if (nx < 0 || nx >= 28 || ny < 0 || ny >= 28) continue;

			// Normalized distance from center
			double d = std::sqrt(std::pow(i, 2) + std::pow(j, 2)) / brushSize;
			double intensity = 0;
			if (d < 1) {
				// Apply cos^2(d * pi/2)
				intensity = std::pow(std::cos(d * 3.1415926536 / 2), 2);
			}
			double& pixel = img[28 * ny + nx][0];
			// Clamp between 0-1
			pixel = std::max(0.0, std::min(1.0, pixel + (white * intensity)));
		}
	}
}

int main() {
	try {
		// Load network data from `nn.dat`
		NeuralNetwork nn;
		std::ifstream in("./nn.dat", std::ios::binary);
		if (in.good()) nn.load(in);
		else std::cerr << "Could not open network data file";
		in.close();

		// Create a window
		sf::RenderWindow window(sf::VideoMode({ 750,500 }), "NNPaint");

		// Load Arial font
		// Note: openFromFile() doesn't seem to find the file even if it exists
		// So, the font file is loaded into a std::vector<char> buffer via std::ifstream
		// Then, the font is loaded from that buffer memory
		std::ifstream file(fontPath, std::ios::binary);
		std::vector<char> buffer((std::istreambuf_iterator<char>(file)), {});
		sf::Font font;
		if (!font.openFromMemory(buffer.data(), buffer.size())) {
			std::cerr << "Failed to load font from memory.\n";
			return 1;
		}
		std::cout << "Font loaded successfully!\n";

		while (window.isOpen()) {
			while (const std::optional event = window.pollEvent()) {
				if (event->is<sf::Event::Closed>())
					window.close();
			}
			window.clear();
			// Left Mouse: White brush
			if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
				draw(window, 1);
			// Right Mouse: Black brush/Erase
			if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right))
				draw(window, -1);
			// 'C' Key: Clear canvas
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Scan::C))
				for (int i = 0; i < 28 * 28; i++)
					img[i][0] = 0;
			// Display canvas
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					sf::RectangleShape square({ squareSize, squareSize });
					int intensity = static_cast<int>(img[28 * j + i][0] * 255);
					square.setFillColor(sf::Color(intensity, intensity, intensity));
					square.setPosition({ static_cast<float>(squareSize * i), static_cast<float>(squareSize * j) });
					square.setOutlineColor(sf::Color(127, 127, 127));
					square.setOutlineThickness(1);
					window.draw(square);
				}
			}
			// Display prediction
			NNMatrix res = nn.run(img);
			for (int i = 0; i < 10; i++) {
				sf::CircleShape circle;
				circle.setRadius(20.f);
				circle.setPosition({ static_cast<float>(580),static_cast<float>(25 + 40 * i) });
				int intensity = static_cast<int>(res[i][0] * 255);
				sf::Color opp = (intensity > 128) ? sf::Color::Black : sf::Color::White;
				circle.setFillColor(sf::Color(intensity, intensity, intensity));
				circle.setOutlineColor(opp);
				circle.setOutlineThickness(1);
				window.draw(circle);

				sf::Text label(font);
				// Note: using a string directly in setString() throws a memory runtime error
				// So, it is converted into a C string (const char*)
				label.setString(std::to_string(i).c_str());
				label.setCharacterSize(20); // in pixels
				label.setFillColor(opp);
				label.setPosition({ static_cast<float>(595),static_cast<float>(32 + 40 * i) });
				window.draw(label);

				sf::Text confidence(font);
				std::string val = std::to_string(res[i][0] * 100) + "%";
				confidence.setString(val.c_str()); // Convert to C string
				confidence.setCharacterSize(20); // in pixels
				int halfIntensity = 128 + intensity / 2;
				confidence.setFillColor(sf::Color(halfIntensity, halfIntensity, halfIntensity));
				confidence.setPosition({ static_cast<float>(630),static_cast<float>(32 + 40 * i) });
				window.draw(confidence);
			}
			window.display();
		}
	}
	catch (sf::Exception& e) {
		std::cout << "Exception: " << e.what();
	}
}
