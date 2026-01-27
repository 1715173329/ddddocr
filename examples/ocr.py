import ddddocr

with open("../samples/yzm2.jpeg", "rb") as f:
    data = f.read()

ocr = ddddocr.DdddOcr(det=True, ocr=False)

result = ocr.detection(data)

print(result)