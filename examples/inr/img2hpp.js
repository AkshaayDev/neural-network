// This script converts an image to a .hpp file containing the pixel data as a flattened vector of grayscale values
const sharp = require("sharp");
const fs = require("fs");

const input = "./img/img.png";
const output = "./img/img.hpp";

(async () => {
	try {
		// Read the image
		const image = await sharp(input)
			.greyscale() // Convert to grayscale
			.raw() // Get raw pixel data
			.toBuffer(); // Convert to buffer

		// Get the image metadata (width and height)
		const metadata = await sharp(input).metadata();
		const width = metadata.width;
		const height = metadata.height;

		// Create a string to hold the contents of the text file
		let contents = "#include <vector>\n";
		contents += `const int width = ${width}, height = ${height};\n`;
		contents += "const std::vector<double> expected = {";
		for (let i = 0; i < image.length; i++) {
			// Add a newline and tab before each line
			if (i % width === 0) contents += "\n\t";
			contents += (image[i] / 255).toFixed(6);
			// Add a comma after each value except the last one
			if (i != image.length - 1) contents += ",";
		}
		contents += "\n};\n";
		fs.writeFileSync(output, contents);

		console.log(`Image successfully processed and saved to ${output}`);
	} catch (err) {
		console.error("Error processing image:", err);
	}
})();
