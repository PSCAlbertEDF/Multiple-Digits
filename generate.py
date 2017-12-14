#coding=utf-8
import random
import string
from PIL import Image,ImageDraw,ImageFont,ImageFilter

font_path = '/Library/Fonts/Arial.ttf'

#生成几位数的验证码
number = 9
#生成验证码图片的高度和宽度
size = (165,35)
#背景颜色，默认为白色
bgcolor = (139,69,19)
#字体颜色，默认为蓝色
fontcolor = (0,0,0)
#干扰线颜色。默认为红色
linecolor = (255,0,0)
#是否要加入干扰线
draw_line = False
#加入干扰线条数的上下限
line_number = (1,5)

#用来随机生成一个字符串
def gene_text():
    source_l = list(string.uppercase)
    source_n = []
    for index in range(0,10):
        source_n.append(str(index))
    st = ' ' + ''.join(random.sample(source_n,1)) + ''.join(random.sample(source_l,3)) + ''.join(random.sample(source_n,3)) + ''.join(random.sample(source_l,2))
    return st

#用来绘制干扰线
def gene_line(draw,width,height):
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill = linecolor)

#生成验证码
def gene_code():
    width,height = size #宽和高
    image = Image.new('RGBA',(width,height),bgcolor) #创建图片

    #验证码的字体
    font = ImageFont.truetype(font_path,25)
    draw = ImageDraw.Draw(image)  #创建画笔

    #生成字符串
    text = gene_text()
    font_width, font_height = font.getsize(text)
    draw.text(((width - font_width) / number, (height - font_height) / number),text,\
            font= font,fill=fontcolor) #填充字符串
    if draw_line:
        gene_line(draw,width,height)
    #创建扭曲
    image = image.transform((width+20,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)

    #滤镜，边界加强
    # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    #保存验证码图片
    image.save('idencode.png')
if __name__ == "__main__":
    gene_code()