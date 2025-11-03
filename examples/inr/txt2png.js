// This script converts a .txt file containing pixel data into a .png image
// The name of the file (without .txt) is passed as a command line argument
const fs = require("fs");
const sharp = require("sharp");

// Run with `node display.js <name>`
const name = process.argv[2];
const contents = fs.readFileSync(name+".txt", "utf8");

// Parse the pixel data into a 2D array
const pixels = contents.split("\n").map(line => line
	.split(",")
	.filter(val => val.trim() !== "") // Remove trailing commas
	.map(val => {
		const num = parseFloat(val);
		// Clamp and scale from 0-1 to 0-255
		return Math.max(0, Math.min(255, Math.round(num * 255)));
	})
);

// Flatten the 2D pixel array into a 1D pixel buffer
const buffer = Buffer.from(pixels.flat());

// Generate the image
sharp(buffer, {
	raw: {
		width: pixels.length,
		height: pixels[0].length,
		channels: 1, // grayscale
	}
})
.png()
.toFile(name+".png")
.then(() => {
	console.log(`Image sucessfully saved as ${name}.png`);
	// Delete the text file after generating the image
	fs.unlink(name+".txt", (err) => {
		if (err) console.error("Error deleting file:", err);
		else console.log("File deleted successfully");
	});
})
.catch(err => {
	console.error("Error generating image:", err);
});
