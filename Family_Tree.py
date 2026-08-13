import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# 1. قراءة وتحليل ملف البيانات من القرص E
csv_path = r"E:\شجرة العائلة\FamilyTree.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"لم يتم العثور على ملف الـ CSV في: {csv_path}")

df = pd.read_csv(csv_path, encoding='windows-1256', on_bad_lines='skip')

id_col, name_col, father_col, gen_col, branch_col = 'PersonID', 'FullName', 'FatherID', 'Generation', 'Branch'
df = df.dropna(subset=[id_col, name_col])
df[id_col] = df[id_col].astype(int)
df[father_col] = pd.to_numeric(df[father_col], errors='coerce').fillna(0).astype(int)
df[gen_col] = pd.to_numeric(df[gen_col], errors='coerce').fillna(1).astype(int)

# فرز الجذع العمودي (النسب العلوي وصولاً إلى بدري فقط)
trunk_keywords = ['النسب العلوي', 'الجذر', 'أجداد العشيرة']
is_trunk = df[branch_col].astype(str).str.contains('|'.join(trunk_keywords)) | (df[id_col] == 1)
df_trunk = df[is_trunk].sort_values(by=gen_col, ascending=True)  # مرتب من الأقدم للأحدث (تصاعدي للأعلى)
df_branches = df[~is_trunk]

# أبعاد اللوحة الفخمة
WIDTH, HEIGHT = 9500, 8000
img = Image.new("RGB", (WIDTH, HEIGHT), (253, 251, 244))  # خلفية ورقية ملكية
draw = ImageDraw.Draw(img)

# إعداد الخطوط بأحجام ضخمة جداً لسهولة القراءة والوضوح
try:
    font_name = "arial.ttf"
    font_title = ImageFont.truetype(font_name, 44)  # خط ضخم ومناسب لاسم واحد في الخلية
    font_leaf = ImageFont.truetype(font_name, 36)  # للأوراق والفروع
except IOError:
    font_title = font_leaf = ImageFont.load_default()


def fix_text(text):
    if pd.isna(text): return ""
    # تنظيف الاسم من أي أقواس زائدة
    text_clean = str(text).split('(')[0].strip()
    return get_display(reshape(text_clean))


# 2. رسم الجذع الفني (عمود مركزي يحمل خلايا بأسماء فردية)
positions = {}
trunk_x = WIDTH // 2
start_y = HEIGHT - 500
y_step = 240

# رسم طبقات الجذع الشجري المتدرج بالخلفية
for w in range(120, 0, -10):
    color_val = 101 - (w // 3)
    draw.line([(trunk_x, start_y + 150), (trunk_x, 350)], fill=(color_val, 67 - (w // 5), 33), width=w)

# رسم قاعدة الجذع العريضة في الأسفل
draw.polygon([(trunk_x - 400, HEIGHT), (trunk_x + 400, HEIGHT), (trunk_x + 70, start_y), (trunk_x - 70, start_y)],
             fill=(85, 55, 25))

# وضع الأسماء الفردية داخل خلايا الجذع (اسم واحد فقط في كل خلية)
last_trunk_y = start_y
for idx, row in df_trunk.iterrows():
    p_id = int(row[id_col])
    positions[p_id] = (trunk_x, start_y)

    # جلب الاسم كما هو مكتوب في الـ CSV تماماً (اسم واحد/ثنائي دون تراكم)
    single_name = row[name_col]

    # رسم بطاقة الجذع (إطار دائري ملكي فخم يتسع للاسم الواحد بوضوح شديد)
    draw.ellipse([trunk_x - 240, start_y - 65, trunk_x + 240, start_y + 65], fill=(240, 230, 200),
                 outline=(212, 175, 55), width=5)
    draw.ellipse([trunk_x - 230, start_y - 55, trunk_x + 230, start_y + 55], fill=(212, 175, 55, 40),
                 outline=(139, 69, 19), width=2)

    # كتابة الاسم الفردي داخل الخلية
    draw.text((trunk_x, start_y), fix_text(single_name), fill=(40, 20, 0), font=font_title, anchor="mm")

    last_trunk_y = start_y
    start_y -= y_step

# 3. توزيع المناطق الجغرافية للفروع الأربعة الكبرى من فوق "بدري"
branch_regions = {
    'البو عرموش': {'x_start': 600, 'x_end': WIDTH // 2 - 1300},
    'البو حمزة': {'x_start': WIDTH // 2 - 1200, 'x_end': WIDTH // 2 - 450},
    'البو عساف': {'x_start': WIDTH // 2 - 1200, 'x_end': WIDTH // 2 - 450},
    'البو محمد': {'x_start': WIDTH // 2 - 400, 'x_end': WIDTH // 2 + 400},  # البو محمد في الوسط تماماً
    'البو عبد الله': {'x_start': WIDTH // 2 + 500, 'x_end': WIDTH - 600}
}

BRANCH_COLORS = {
    'البو محمد': {'bg': (34, 112, 63), 'text': (255, 255, 255)},
    'البو عرموش': {'bg': (184, 134, 11), 'text': (255, 255, 255)},
    'البو عبد الله': {'bg': (70, 130, 180), 'text': (255, 255, 255)},
    'البو حمزة': {'bg': (128, 0, 32), 'text': (255, 255, 255)},
    'البو عساف': {'bg': (128, 0, 32), 'text': (255, 255, 255)},
    'default': {'bg': (100, 100, 100), 'text': (255, 255, 255)}
}

tree_dict = {}
for _, row in df_branches.iterrows():
    f_id = int(row[father_col])
    c_id = int(row[id_col])
    if f_id not in tree_dict: tree_dict[f_id] = []
    tree_dict[f_id].append(c_id)


def assign_positions(person_id, x_min, x_max, current_y):
    my_x = x_min + (x_max - x_min) // 2
    positions[person_id] = (my_x, current_y)
    if person_id in tree_dict:
        children = tree_dict[person_id]
        c_count = len(children)
        if c_count > 0:
            x_splits = np.linspace(x_min, x_max, c_count + 1)
            for i, child_id in enumerate(children):
                assign_positions(child_id, int(x_splits[i]), int(x_splits[i + 1]), current_y - 340)


if 1 in tree_dict:
    for main_child_id in tree_dict[1]:
        child_rows = df_branches[df_branches[id_col] == main_child_id]
        if len(child_rows) > 0:
            b_name = child_rows.iloc[0][branch_col]
            region = branch_regions.get(b_name, {'x_start': 1000, 'x_end': WIDTH - 1000})
            assign_positions(main_child_id, region['x_start'], region['x_end'], last_trunk_y - 380)


# 4. رسم الأغصان المنحنية الانسيابية الكبرى
def draw_artistic_branch(draw, p1, p2, color, width):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = x1, y1 - (y1 - y2) * 0.4
    points = []
    for t in np.linspace(0, 1, 30):
        x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
        y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
        points.append((x, y))
    draw.line(points, fill=color, width=width, joint="curve")


# الأذرع الكبرى الخارجة من بدري للأولاد الأربعة
if 1 in tree_dict:
    for main_child_id in tree_dict[1]:
        if main_child_id in positions:
            draw_artistic_branch(draw, positions[1], positions[main_child_id], (184, 134, 11), width=14)

# رسم باقي الأغصان الفرعية للأوراق
for _, row in df_branches.iterrows():
    c_id, f_id = int(row[id_col]), int(row[father_col])
    if c_id in positions and f_id in positions and f_id != 1:
        draw_artistic_branch(draw, positions[f_id], positions[c_id], (140, 100, 60), width=5)

# 5. رسم بطاقات الأسماء للأفرع والأوراق
for _, row in df_branches.iterrows():
    c_id = int(row[id_col])
    if c_id in positions and c_id != 1:
        cx, cy = positions[c_id]
        b_name = row[branch_col]
        color_style = BRANCH_COLORS.get(b_name, BRANCH_COLORS['default'])

        # رسم بطاقات الأسماء للأوراق
        draw.rounded_rectangle([cx - 200, cy - 55, cx + 200, cy + 55], radius=18, fill=color_style['bg'],
                               outline=(218, 165, 32), width=3)
        draw.text((cx, cy), fix_text(row[name_col]), fill=color_style['text'], font=font_leaf, anchor="mm")

# 6. حفظ الصورة النهائية مباشرة في المجلد المخصص بالقرص E
output_dir = r"E:\شجرة العائلة"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_image_name = os.path.join(output_dir, "Family_Tree_Artistic_Output.png")
img.save(output_image_name, "PNG", optimize=True)

print(f"✨ تم التحديث بنجاح! اسم واحد نقي داخل كل خلية في الجذع الرئيسي.")
print(f"📁 تفضل بمعاينة الصورة النهائية في مجلدك: {output_image_name}")