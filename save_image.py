
def html_to_int(code):
    return int("0x"+code[1::],16)*256+255

def save_mask(mask,fname):
    colors = [
        "#2E2EFE",
        "#00FF00",
        "#FF0040",
        "#610B0B",
        "#FF0040",
        "#00FFFF",
        "#FFFF00",
        "#848484",
        "#000000",
    ]
    color_ints = [html_to_int(col) for col in colors]
    resarray = np.zeros((IMAGE_WIDTH,IMAGE_WIDTH),dtype=np.int32)
    for x in range(IMAGE_WIDTH):
        for y in range(IMAGE_WIDTH):
            sum = 0
            iters = 0
            for z in range(DROPOUT_CHANNELS):
                if mask[x][y][z] != 0:
                    sum += color_ints[z]
                    iters += 1
            resarray[x][y] += sum // iters if iters != 0 else 0xffffffff

    casted_image = np.frombuffer(resarray.tobytes(), dtype=np.uint8).reshape((IMAGE_WIDTH,IMAGE_WIDTH,4))
    Image.fromarray(casted_image,mode="RGBA").save(fname)



def save_generated_image(orig_img,generated_image,generated_mask,revealed_capsules,weight_updates):
    folder = "generated/{}/".format(str(weight_updates))
    NUM_IMAGES_SAVE = 5
    if not os.path.exists(folder):
        for i in range(NUM_IMAGES_SAVE):
            subfold = folder+str(i)+"/"
            fname = subfold+"orig.png"
            os.makedirs(subfold)
            save_image(orig_img[i].reshape((IMAGE_WIDTH,IMAGE_WIDTH)),fname)
    for i in range(NUM_IMAGES_SAVE):
        fname = "{}{}/{}.png".format(folder,i,revealed_capsules)
        maskfname = "{}{}/{}m.png".format(folder,i,revealed_capsules)
        save_image(generated_image[i].reshape((IMAGE_WIDTH,IMAGE_WIDTH)),fname)
        save_mask(generated_mask[i],maskfname)
