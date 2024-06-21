import os
import sys
import glob
import cv2
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
sys.path.append('../lib')
from align import align_nocs_to_depth
from utils import load_depth


def create_img_list(data_dir):
    """ Create train/val/test data list for CAMERA and Real. 
    主要功能是为给定数据目录下的CAMERA和Real数据集创建训练、验证和测试的图像路径列表。
    """
    # CAMERA dataset
    # 创建两个数据集


    # 在CAMERA数据集中，训练集和测试集的文件夹是'train'和'val'
    for subset in ['train', 'val']:
        img_list = []
        img_dir = os.path.join(data_dir, 'CAMERA', subset)
        folder_list = [name for name in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, name))]
        for i in range(10*len(folder_list)):
            folder_id = int(i) // 10
            img_id = int(i) % 10
            img_path = os.path.join(subset, '{:05d}'.format(folder_id), '{:04d}'.format(img_id))
            img_list.append(img_path)
        with open(os.path.join(data_dir, 'CAMERA', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)

    # Real dataset
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, 'Real', subset)
        # os.listdir(img_dir)：这个函数会返回img_dir目录下所有的文件和目录的名称列表。
        # os.path.isdir(os.path.join(img_dir, name))：这部分代码检查每一个name是否是一个目录。
        # os.path.join(img_dir, name)会生成完整路径，然后os.path.isdir()判断该路径是否指向一个目录。
        # [name for name in sorted(...) if ...]：这是一个列表推导式（List Comprehension），
        # 它从sorted(os.listdir(img_dir))得到排序后的文件和目录列表，然后通过if os.path.isdir(...)条件过滤，只保留那些确实是目录的name。
        # 整个列表推导式的作用就是获取img_dir目录下所有 ！子目录！ 的名称，并且这些名称是按字母顺序排列的。
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            # 用于返回所有匹配给定的文件路径模式的文件列表。
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            # 这行代码的作用是将列表 img_paths 中的所有元素（在这个上下文中，这些元素是文件路径）按照字母顺序进行排序。
            img_paths = sorted(img_paths)

            # 
            for img_full_path in img_paths:
                # 只返回最后的文件名，包括文件扩展名。
                img_name = os.path.basename(img_full_path)
                # img_ind = img_name.split('_')[0] 这行代码的作用是从文件名中提取出以下划线 _ 分隔的第一个部分。
                # split('_') 方法会将 img_name 字符串分割成多个部分，每个部分都是由下划线 _ 分隔开的子字符串。结果是一个列表，其中包含了所有的子字符串。
                img_ind = img_name.split('_')[0]
                # 去掉后面的_color.png
                
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        # 创建文件
        with open(os.path.join(data_dir, 'Real', subset+'_list_all.txt'), 'w') as f:
            for img_path in img_list:
                # 表示一个字符串，% img_path 则是将 img_path 的值插入到字符串中。
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')


def process_data(img_path, depth):
    """ Load instance masks for the objects in the image. 
    处理单张照片的数据
    masks:
    维度: (h, w, i)
    意义: masks是一个三维数组，其中h和w分别是图像的高度和宽度，i是有效处理的实例数量。
    每个实例都有一个对应的二值掩码，表示该实例在图像中的位置。具体来说，masks[:, :, j]是第j个实例的掩码，其中True（或1）的值表示实例的像素位置，False（或0）表示背景或其他实例的像素位置。
    
    coords:维度: (h, w, i, 3)
    意义: coords是一个四维数组，其中前三维与masks相同，第四维3表示每个像素的三维坐标信息。coords[:, :, j, :]包含了第j个实例的像素坐标信息，其中每个像素的坐标值被存储为一个三元组，通常表示像素在三维空间中的位置（例如，相对于相机的位置）。

    class_ids:
        类型: 列表
        意义: 包含每个实例所属的类别ID。class_ids[j]是第j个实例的类别ID，用于识别和分类不同的对象类型。

    instance_ids:
        类型: 列表
        意义: 包含每个实例的唯一ID。instance_ids[j]是第j个实例的ID，用于唯一标识图像中的每个对象实例。

    model_list:
        类型: 列表
        意义: 包含每个实例对应的3D模型ID或名称。model_list[j]是第j个实例的3D模型信息，这在进行3D重建或与3D模型数据库关联时非常有用。
bboxes:
    维度: (i, 4)
    意义: bboxes是一个二维数组，其中i是有效处理的实例数量，4表示每个实例的边界框坐标。bboxes[j, :]包含了第j个实例的边界框坐标，通常格式为 [y1, x1, y2, x2]，其中(x1, y1)是边界框的左上角坐标，(x2, y2)是边界框的右下角坐标。

    """
    mask_path = img_path + '_mask.png'
    # 只取第三个通道。可能是掩码每个通道都差不多
    mask = cv2.imread(mask_path)[:, :, 2]
    mask = np.array(mask, dtype=np.int32)
    # np.unique(mask) 返回一个数组，其中包含了mask中出现的所有唯一数值。np.unique(mask)本来就排过顺序了
    # all_inst_ids，把所有出现过的数值排个顺序。list转换为Python的列表。sorted也是排序，但是前面已经排过了，没什么用
    all_inst_ids = sorted(list(np.unique(mask)))

    # 检查数据格式，如果all_inst_ids最后一个元素不是255，则抛出异常
    assert all_inst_ids[-1] == 255
    del all_inst_ids[-1]    # remove background 移出背景序号。
    # 出现了几个标号
    num_all_inst = len(all_inst_ids)
    # 深度图的高和宽，cv2读出来的都是高在前，宽在后
    h, w = mask.shape

    # 读取_coord.png
    coord_path = img_path + '_coord.png'
    # 0,1,2不包括3，但是其实本身也是三维的，所以:3没什么用。
    coord_map = cv2.imread(coord_path)[:, :, :3]
    # 重新排列顺序 OpenCV 默认读取的图像通常是 BGR 格式，而许多其他图像处理库和显示设备期望的是 RGB 格式。
    coord_map = coord_map[:, :, (2, 1, 0)]
    # flip z axis of coord map z方向归一化处理 三维坐标系中反转Z轴的方向
    coord_map = np.array(coord_map, dtype=np.float32) / 255
    # 可能原始数据集近大远小现在换成了近小远大，或者反过来
    coord_map[:, :, 2] = 1 - coord_map[:, :, 2]

    class_ids = []
    instance_ids = []
    model_list = []
    # 初始化mask 坐标系，框框
    masks = np.zeros([h, w, num_all_inst], dtype=np.uint8)
    coords = np.zeros((h, w, num_all_inst, 3), dtype=np.float32)
    bboxes = np.zeros((num_all_inst, 4), dtype=np.int32)

    # 取出这个框
    meta_path = img_path + '_meta.txt'
    with open(meta_path, 'r') as f:
        i = 0
        # 遍历f中的每一行，为line 这个是遍历每一幅图的信息
        for line in f:
            # strip()去除行尾的换行符，split(' ')按空格分隔每一项，获取每个实例的基本信息。
            line_info = line.strip().split(' ')
            # 存储实例ID。
            inst_id = int(line_info[0])
            cls_id = int(line_info[1])
            # background objects and non-existing objects
            if cls_id == 0 or (inst_id not in all_inst_ids):
                continue
            if len(line_info) == 3:
                # Real数据集
                model_id = line_info[2]    # Real scanned objs
            else:
                model_id = line_info[3]    # CAMERA objs
            # remove one mug instance in CAMERA train due to improper model
            if model_id == 'b9be7cfe653740eb7633a2dd89cec754':
                continue
            # process foreground objects，NumPy会尝试通过广播机制使它们具有相同的形状。分离出每一个标签的mask
            inst_mask = np.equal(mask, inst_id) 
            # bounding box np.any()函数沿着指定的轴（axis）检查数组中是否存在任何一个元素为True。
            # 在这个例子中，axis=0意味着函数将在垂直方向上进行操作，即对于inst_mask的每一列，np.any()会检查这一列中是否至少有一个元素为True。
            horizontal_indicies = np.where(np.any(inst_mask, axis=0))[0]
            # 变量将包含inst_mask中所有非零列的索引，这些索引实际上就是实例在水平方向上的投影，可以用来确定实例的左右边界。(水平边界，从左边开始
            # 在实际应用中，这通常会用于计算实例的边界框，即x1和x2坐标，这两个坐标分别代表实例在图像中最左侧和最右侧的位置。
            # 垂直边界从上面开始
            vertical_indicies = np.where(np.any(inst_mask, axis=1))[0]
            
            assert horizontal_indicies.shape[0], print(img_path)
            # x1和x2分别代表最左侧和最右侧
            x1, x2 = horizontal_indicies[[0, -1]]
            # 最上面和最下面
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            # object occupies full image, rendering error, happens in CAMERA dataset 只要超出尺寸了，就返回。
            if np.any(np.logical_or((x2-x1) > 600, (y2-y1) > 440)):
                return None, None, None, None, None, None
            # not enough valid depth observation
            final_mask = np.logical_and(inst_mask, depth > 0)
            if np.sum(final_mask) < 64:
                continue
            # cls_id是信息中的第二列
            class_ids.append(cls_id)
            # inst_id是inst的顺序
            instance_ids.append(inst_id)
            # model_id是个字符串
            model_list.append(model_id)
            masks[:, :, i] = inst_mask
            # 将特定的序号的坐标保留下来，其它的全变成0
            coords[:, :, i, :] = np.multiply(coord_map, np.expand_dims(inst_mask, axis=-1))
            # bboxes是矩形框
            bboxes[i] = np.array([y1, x1, y2, x2])
            i += 1
    # no valid foreground objects
    if i == 0:
        return None, None, None, None, None, None


    # masks数组最初被初始化为一个固定大小的三维数组，其中第三维的大小等于所有潜在实例的数量（num_all_inst）。
    # 然而，经过处理，可能有些实例没有被有效检测或满足条件，导致i（有效实例计数器）小于num_all_inst。
    # 这行代码通过切片操作[:, :, :i]将masks数组的大小调整为只包含有效实例的掩码。
    # 这样，输出的masks数组大小就会与实际处理的实例数量一致，避免了不必要的空间占用。
    # 前两个维度是横纵坐标，最后一个是 inst_id
    masks = masks[:, :, :i]
    # clip确保在 01范围内
    coords = np.clip(coords[:, :, :i, :], 0, 1)
    bboxes = bboxes[:i, :]

    return masks, coords, class_ids, instance_ids, model_list, bboxes


def annotate_camera_train(data_dir):
    """ Generate gt labels for CAMERA train data. """
    camera_train = open(os.path.join(data_dir, 'CAMERA', 'train_list_all.txt')).read().splitlines()
    intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    for img_path in tqdm(camera_train):
        img_full_path = os.path.join(data_dir, 'CAMERA', img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        if not all_exist:
            continue
        depth = load_depth(img_full_path)
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # Umeyama alignment of GT NOCS map with depth image
        scales, rotations, translations, error_messages, _ = \
            align_nocs_to_depth(masks, coords, depth, intrinsics, instance_ids, img_path)
        if error_messages:
            continue
        # re-label for mug category
        for i in range(len(class_ids)):
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = translations[i] - scales[i] * rotations[i] @ T0
                s = scales[i] / s0
                scales[i] = s
                translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(data_dir, 'CAMERA/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


def annotate_real_train(data_dir):
    """ Generate gt labels for Real train data through PnP. """

    # read() 方法用于读取文件的全部内容。这会将文件中的所有文本读入到一个大的字符串中。
    # splitlines() 方法用于将这个大字符串按照行分割成多个小字符串，并返回一个列表。
    # splitlines() 方法会自动识别换行符（\n 或 \r\n 等），并根据它们将字符串分割成多行。
    # 这样就得到了一个列表，其中每个元素都是文件中原来的一行。
    real_train = open(os.path.join(data_dir, 'Real/train_list_all.txt')).read().splitlines()

    # 相机内参矩阵
    # [
    # K =
    # \begin{bmatrix}
    # f_x & 0 & c_x \
    # 0 & f_y & c_y \
    # 0 & 0 & 1
    # \end{bmatrix}
    # fx​：这是在图像的x轴方向上的焦距，以像素为单位。它描述了镜头的放大率在x方向上的效果。在物理相机中，焦距是以毫米为单位的，但在计算机视觉中，为了方便，我们将其转换为像素单位。
    # fyfy​：这是在图像的y轴方向上的焦距，同样以像素为单位。对于大多数现代相机，fxfx​ 和 fyfy​ 往往非常接近，但由于传感器像素的非正方形性或镜头设计的不同，它们可能不完全相同。
    # cxcx​：这是图像的光心（principal point）在x轴方向的位置，以像素为单位。
    # 光心是指光线通过镜头后汇聚的点在图像平面上的投影位置。在理想情况下，光心应该位于图像的中心，但实际中可能会有偏差。
    # cycy​：这是图像的光心在y轴方向的位置，同样以像素为单位。
    # ]
    intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    
    # scale factors for all instances
    scale_factors = {}

    # 3D对象边界框尺寸信息


    path_to_size = glob.glob(os.path.join(data_dir, 'obj_models/real_train', '*_norm.txt'))
    # glob.glob()：这是glob模块提供的函数，用于查找与给定模式匹配的所有文件路径。这些文件大概都是标定的位置
    # 它接收一个字符串参数，该参数描述了要匹配的文件模式。glob.glob()函数会返回一个包含所有匹配文件路径的列表。

    # sorted(path_to_size)给这个列表排个序
    # 3D对象边界框尺寸信息
    for inst_path in sorted(path_to_size):
        # 把文件扩展名去掉 basename 是取了最后一个斜线，也就是取出文件名再把扩展名去掉，这个是作为一个字典在用
        instance = os.path.basename(inst_path).split('.')[0]
        # 加载这个文件
        bbox_dims = np.loadtxt(inst_path)
        # scale_factors[instance] = np.linalg.norm(bbox_dims)：此行为每个实例计算尺度因子，并将其存储在字典 scale_factors 中。
        # 尺度因子是通过计算 bbox_dims 的范数（norm）得到的。
        # 计算得到的尺度因子存储在字典 scale_factors 中，以实例名称作为键。
        # instance是一个字典。
        scale_factors[instance] = np.linalg.norm(bbox_dims)
    # meta info for re-label mug category
    # 这里以二进制读取模式（'rb'）打开文件，这是因为pickle模块通常使用二进制格式来保存对象，以提高效率和兼容性。
    # mug_meta.pkl文件通常包含与水杯类别相关的元数据，这些数据可能是在数据预处理阶段收集和保存的，用于后续的数据分析或模型训练。
    # 应该就是mug_handle.pkl这个文件
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    valid_img_list = []
    # 添加进度条 img_path就是txt文件列表中列出来的每一行的每个文件(train_list_all)
    for img_path in tqdm(real_train):
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        # _color
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_coord.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_mask.png') and \
                    os.path.exists(img_full_path + '_meta.txt')
        # 通过将这些 os.path.exists() 调用连接在一起，并使用逻辑与运算符 and，代码确保了只有当所有五种类型的文件都存在时，all_exist 变量才会被设置为 True。如果任何一个文件缺失，all_exist 将为 False。
        if not all_exist:
            continue
        # 读取了一个深度图的矩阵，大概分辨率为(480, 640)
        depth = load_depth(img_full_path)
        # 数据处理
        masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
        if instance_ids is None:
            continue
        # compute pose
        # 实例数量，因为比如类别id为 4,1,1,2,3,虽然有4个类别，但是这个长度是5
        num_insts = len(class_ids)
        # scales初始化了一个这样长度的零向量
        scales = np.zeros(num_insts)
        # num_insts个3x3的旋转矩阵
        rotations = np.zeros((num_insts, 3, 3))
        # num_insts个位置向量
        translations = np.zeros((num_insts, 3))

        # 这个循环迭代图像中的每个实例，num_insts是处理图像中实例的数量。
        for i in range(num_insts):
            # 这里获取了第i个实例对应的3D模型的缩放因子。
            s = scale_factors[model_list[i]]
            mask = masks[:, :, i]
            # 使用np.where()函数找到掩码中所有非零（即属于实例）像素的位置索引。
            # idxs是一个元组，包含两组索引：第一个数组是行索引（y坐标），第二个数组是列索引（x坐标）。
            # idxs的数值都是成对出现，然后每一个[0]和[1]中的数值都是一一对应的，代表着每一个坐标，而且每一个元组的数值都是可以重复的
            idxs = np.where(mask)
            # 这里从coords数组中选取第i个实例的坐标信息。coords是一个四维数组，其中第三维表示实例，第四维表示每个像素的3D坐标。coord现在是一个三维数组，存储了第i个实例的每个像素的3D坐标。
            coord = coords[:, :, i, :]
            # coords中的值在0到1之间
            coord_pts = s * (coord[idxs[0], idxs[1], :] - 0.5)
            # coord_pts[:, :, None]是在第三维后面增加一个维度转化为四维数组
            coord_pts = coord_pts[:, :, None]
            # 调换一下顺序，调换成x，y，然后再转置一下
            img_pts = np.array([idxs[1], idxs[0]]).transpose()
            # 二维加了一维，是在第三维加了一维度，和之前的操作类似，但是调用的方法不一样。
            img_pts = img_pts[:, :, None].astype(float)
            # 畸变系数 
            distCoeffs = np.zeros((4, 1))    # no distoration
            # retval代表是否解算成功 rvec代表轴角，tvec代表平移向量 
            retval, rvec, tvec = cv2.solvePnP(coord_pts, img_pts, intrinsics, distCoeffs)
            # 如果解算失败，抛出异常
            assert retval
            # 轴角转化为旋转矩阵，_是一个可选的雅可比矩阵
            R, _ = cv2.Rodrigues(rvec)
            # 这里的tvec通常是一个形状为(1, 3)或(3, 1)的一维向量，表示平移向量。np.squeeze(tvec)会检查tvec的形状，并移除所有长度为1的维度。
            # 把维度为1的维度去掉
            T = np.squeeze(tvec)
            # re-label for mug category
            if class_ids[i] == 6:
                T0 = mug_meta[model_list[i]][0]
                s0 = mug_meta[model_list[i]][1]
                T = T - s * R @ T0
                s = s / s0
            scales[i] = s
            rotations[i] = R
            translations[i] = T
        # write results
        gts = {}
        gts['class_ids'] = class_ids    # int list, 1 to 6
        gts['bboxes'] = bboxes  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = scales.astype(np.float32)  # np.array, scale factor from NOCS model to depth observation
        gts['rotations'] = rotations.astype(np.float32)    # np.array, R
        gts['translations'] = translations.astype(np.float32)  # np.array, T
        gts['instance_ids'] = instance_ids  # int list, start from 1
        gts['model_list'] = model_list  # str list, model id/name

        # b代表二进制，w代表写入
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(os.path.join(data_dir, 'Real/train_list.txt'), 'w') as f:
        for img_path in valid_img_list:
            # % 是个占位符
            f.write("%s\n" % img_path)


def annotate_test_data(data_dir):
    """ Generate gt labels for test data.
        Properly copy handle_visibility provided by NOCS gts.
    """
    # Statistics:
    # test_set    missing file     bad rendering    no (occluded) fg    occlusion (< 64 pts)
    #   val        3792 imgs        132 imgs         1856 (23) imgs      50 insts
    #   test       0 img            0 img            0 img               2 insts

    camera_val = open(os.path.join(data_dir, 'CAMERA', 'val_list_all.txt')).read().splitlines()
    real_test = open(os.path.join(data_dir, 'Real', 'test_list_all.txt')).read().splitlines()
    camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]])
    real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    # compute model size
    # model_file_path = ['obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
    model_file_path = ['obj_models/real_test.pkl']

    models = {}
    for path in model_file_path:
        with open(os.path.join(data_dir, path), 'rb') as f:
            models.update(cPickle.load(f))
    model_sizes = {}
    for key in models.keys():
        model_sizes[key] = 2 * np.amax(np.abs(models[key]), axis=0)
    # meta info for re-label mug category
    with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    # subset_meta = [('CAMERA', camera_val, camera_intrinsics, 'val'), ('Real', real_test, real_intrinsics, 'test')]
    subset_meta = [('Real', real_test, real_intrinsics, 'test')]
    for source, img_list, intrinsics, subset in subset_meta:
        valid_img_list = []
        for img_path in tqdm(img_list):
            img_full_path = os.path.join(data_dir, source, img_path)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_coord.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_mask.png') and \
                        os.path.exists(img_full_path + '_meta.txt')
            if not all_exist:
                continue
            depth = load_depth(img_full_path)
            masks, coords, class_ids, instance_ids, model_list, bboxes = process_data(img_full_path, depth)
            if instance_ids is None:
                continue
            num_insts = len(instance_ids)
            # match each instance with NOCS ground truth to properly assign gt_handle_visibility
            nocs_dir = os.path.join(os.path.dirname(data_dir), 'results/nocs_results')
            if source == 'CAMERA':
                nocs_path = os.path.join(nocs_dir, 'val', 'results_val_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            else:
                nocs_path = os.path.join(nocs_dir, 'real_test', 'results_test_{}_{}.pkl'.format(
                    img_path.split('/')[-2], img_path.split('/')[-1]))
            with open(nocs_path, 'rb') as f:
                nocs = cPickle.load(f)
            gt_class_ids = nocs['gt_class_ids']
            gt_bboxes = nocs['gt_bboxes']
            gt_sRT = nocs['gt_RTs']
            gt_handle_visibility = nocs['gt_handle_visibility']
            map_to_nocs = []
            for i in range(num_insts):
                gt_match = -1
                for j in range(len(gt_class_ids)):
                    if gt_class_ids[j] != class_ids[i]:
                        continue
                    if np.sum(np.abs(bboxes[i] - gt_bboxes[j])) > 5:
                        continue
                    # match found
                    gt_match = j
                    break
                # check match validity
                assert gt_match > -1, print(img_path, instance_ids[i], 'no match for instance')
                assert gt_match not in map_to_nocs, print(img_path, instance_ids[i], 'duplicate match')
                map_to_nocs.append(gt_match)
            # copy from ground truth, re-label for mug category
            handle_visibility = gt_handle_visibility[map_to_nocs]
            sizes = np.zeros((num_insts, 3))
            poses = np.zeros((num_insts, 4, 4))
            scales = np.zeros(num_insts)
            rotations = np.zeros((num_insts, 3, 3))
            translations = np.zeros((num_insts, 3))
            for i in range(num_insts):
                gt_idx = map_to_nocs[i]
                sizes[i] = model_sizes[model_list[i]]
                sRT = gt_sRT[gt_idx]
                s = np.cbrt(np.linalg.det(sRT[:3, :3]))
                R = sRT[:3, :3] / s
                T = sRT[:3, 3]
                # re-label mug category
                if class_ids[i] == 6:
                    T0 = mug_meta[model_list[i]][0]
                    s0 = mug_meta[model_list[i]][1]
                    T = T - s * R @ T0
                    s = s / s0
                # used for test during training
                scales[i] = s
                rotations[i] = R
                translations[i] = T
                # used for evaluation
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = s * R
                sRT[:3, 3] = T
                poses[i] = sRT
            # write results
            gts = {}
            gts['class_ids'] = np.array(class_ids)    # int list, 1 to 6
            gts['bboxes'] = bboxes    # np.array, [[y1, x1, y2, x2], ...]
            gts['instance_ids'] = instance_ids    # int list, start from 1
            gts['model_list'] = model_list    # str list, model id/name
            gts['size'] = sizes   # 3D size of NOCS model
            gts['scales'] = scales.astype(np.float32)    # np.array, scale factor from NOCS model to depth observation
            gts['rotations'] = rotations.astype(np.float32)    # np.array, R
            gts['translations'] = translations.astype(np.float32)    # np.array, T
            gts['poses'] = poses.astype(np.float32)    # np.array
            gts['handle_visibility'] = handle_visibility    # handle visibility of mug
            with open(img_full_path + '_label.pkl', 'wb') as f:
                cPickle.dump(gts, f)
            valid_img_list.append(img_path)
        # write valid img list to file
        with open(os.path.join(data_dir, source, subset+'_list.txt'), 'w') as f:
            for img_path in valid_img_list:
                f.write("%s\n" % img_path)


if __name__ == '__main__':
    data_dir = '/media/wangxiao/新加卷1/NOCS数据集2/data'
    # create list for all data
    # 创建数据集路径，
    create_img_list(data_dir)
    # annotate dataset and re-write valid data to list
    # camera数据集
    annotate_camera_train(data_dir)
    annotate_real_train(data_dir)
    annotate_test_data(data_dir)
