import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

#font = ImageFont.truetype("Arial.ttf",14)
font = ImageFont.truetype("OpenSans-Bold.ttf",50)
#img=Image.new("RGBA", (84,84),(255,255,255))
img=Image.new('L', (84,84),(255))
draw = ImageDraw.Draw(img)
draw.text((0, 10),"1",font=font)
draw = ImageDraw.Draw(img)
img.save("a_test.png")

