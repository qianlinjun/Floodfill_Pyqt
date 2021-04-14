import os
import numpy as np
from pykml import parser
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

path = '{}/templates/'.format(os.path.dirname(__file__))

# 创建一个加载器, jinja2 会从这个目录中加载模板
loader = FileSystemLoader(path)

# 用加载器创建一个环境, 有了它才能读取模板文件
env = Environment(loader=loader)


def gen_kml_from_lat_lon():
    # __file__ 就是本文件的名字
    # 得到放置模板的目录


    template = env.get_template("google_point.j2")  

    # file_path = r"C:\qianlinjun\项目\11-14\山区\swizz-test\0-client.sh"
    # with open(file_path) as f:
    #     for line in f.readlines():
    #         if "LocationCheck" in line:

    idx = 0
    for lat in np.arange(46.60, 47.09, 0.009077*2):
        for lon in np.arange(8.48, 8.81, 0.0137*2):
            # print(lat, lon)
            content = template.render(name=idx, latitude=float(lat), longitude=float(lon))
            with open(r'C:\qianlinjun\graduate\gen_dem\output\img_with_mask\2km_2km\{}.kml'.format(idx),'w') as fp:
                fp.write(content)
            idx += 1

    



def modify_kml_file(kml_path):
    # <Placemark>
	# 	<name>10</name>
	# 	<LookAt>
	# 		<longitude>8.521622549037348</longitude>
	# 		<latitude>46.60557699697863</latitude>
	# 		<altitude>0</altitude>
	# 		<heading>58.82902570249206</heading>
	# 		<tilt>90</tilt>
	# 		<range>200</range>
	# 		<gx:altitudeMode>relativeToSeaFloor</gx:altitudeMode>
	# 	</LookAt>
	# 	<styleUrl>#m_ylw-pushpin21</styleUrl>
	# 	<Point>
	# 		<altitudeMode>relativeToGround</altitudeMode>
	# 		<gx:drawOrder>1</gx:drawOrder>
	# 		<coordinates>8.515921843431936,46.60295295648127,0</coordinates>
	# 	</Point>
	# </Placemark>

    template = env.get_template("google_point-exact.j2")  
    max_id = -1
    with open(kml_path, 'r') as f:
        kml = parser.parse(f).getroot()
        for pt in kml.Document.Placemark: # 遍历所有的Placemark Document.Placemark
            name = pt.name
            if name != "None":
                if int(name) > max_id:
                    max_id = int(name)
        for pt in kml.Document.Placemark: # 遍历所有的Placemark Document.Placemark
            name = pt.name
            print(name,str(pt.Point.coordinates))
            look_longitude = pt.LookAt.longitude
            look_latitude = pt.LookAt.latitude
            look_altitude = pt.LookAt.altitude
            heading = pt.LookAt.heading
            tilt = pt.LookAt.tilt
            range_ = 200
            coods = str(pt.Point.coordinates)
            lon = coods.split(",")[0]
            lat = coods.split(",")[1]
            if name == "None":
                max_id += 1
                name = max_id
            content = template.render(name = name, look_longitude=look_longitude, \
                look_latitude=look_latitude, look_altitude = look_altitude, \
                heading= heading, tilt=tilt, range=range_, latitude=lat, longitude=lon)
            
            with open(r'C:\qianlinjun\graduate\data\switz-test-pts-3-23-fin\{}.kml'.format(name),'w') as fp:
                fp.write(content)
            # idx += 1
        print(len(kml.Document.Placemark)) # 216地标



def create_kml_from_txt(txtDir):
    from lxml import etree  #将KML节点输出为字符串
    import xlrd             #操作Excel
    from pykml.factory import KML_ElementMaker as KML #使用factory模块
    #使用第一个点创建Folder
    fold = KML.Folder(KML.Placemark(
        KML.Point(KML.coordinates('0,0,0'))
        )
    )


    txtPathObj = Path(txtDir)
    for filePath in txtPathObj.iterdir():
        filePath = str(filePath)
        if "png" in filePath:
            # with open(filePath, "r") as txt_f:
            #     txt_conts = txt_f.readlines()
            #     name = txt_conts[6].strip() 
            #     lon = txt_conts[2].strip()
            #     lat = txt_conts[1].strip()
            #     fold.append(KML.Placemark(KML.name(name),
            #     KML.Point(KML.coordinates(str(lon) +','+ str(lat) +',0')))
            #     )
            # lon lat
            file_loc = filePath.split("\\")[-1].split("_")[1:3]
            # file_loc = list(map(float, file_loc))
            fold.append(KML.Placemark(KML.name(filePath.split("\\")[-1].replace(".png", "")),
                KML.Point(KML.coordinates(file_loc[0] +','+ file_loc[1] +',0')))
                )

    #使用etree将KML节点输出为字符串数据
    content = etree.tostring(etree.ElementTree(fold),pretty_print=True)
    # print(content)
    #保存到文件，然后就可以在Google地球中打开了
    with open(r'C:\qianlinjun\graduate\test-data\query\query-auto.kml', 'wb') as fp:
        fp.write(content)



if __name__ == '__main__':
    # modify_kml_file("C:\qianlinjun\graduate\data\switz-test-pts-3-23-fin.kml")
    create_kml_from_txt(r"C:\qianlinjun\graduate\test-data\query")

