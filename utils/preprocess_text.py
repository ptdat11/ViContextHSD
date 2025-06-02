import regex as re
import os
# from py_vncorenlp import VnCoreNLP
# import emoji
import unicodedata
import json

# cwd = os.getcwd()
# vncorenlp = VnCoreNLP(annotators=['wseg'], save_dir=os.environ['VNCORENLP'])
# os.chdir(cwd)

PHONE_RE = re.compile(r"\(?([0-9]{3})\)?([ .-]?)([0-9]{3})\2([0-9]{4})")
DUP_TRALLING_RE = re.compile(r"([\D\w])\1+\b")
SYLLABLE_RE = re.compile(r'(<[^<>]+>)|\b\w+\b|\X')
GROUPING_SEPARATOR_RE = re.compile(r"(?<=\d)[\.,](?=\d{3})")
VALUE_UNIT_RE = re.compile(r"(\d+)(\D+)")
PUNCTUATION_RE = re.compile(
    # "([" + re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~") + "])"
    r'(<[^<>]+>)|([^\w\s])'
)
# EMOJI_RE = emoji.replace_emoji()

SPECIAL_WORD_SUBSTITUTION = {
    "@@": "confuseeyes",
    "℅": "%",
    r"\n": "<newline>",
    r"/": " fraction ",
    r":\)+": "smileface",
    r";\)+": "smileface",
    r":\*+": "kissingface",
    r"=\)+": "playfulsmileface",
    r"=\(+": "playfulsadface",
    r":\(+": "sadface",
    r":3+": "threeface",
    r":v+": "vface",
    r"\^\^": "kindsmile",
    r"\^_\^": "kindmountsmile",
    r"\^\.\^": "kindmountsmile",
    r"-_-": "disapointface",
    r"\._\.": "confusedface",
    r":>+": "cutesmile",
    r"(\|)w(\|)": "fancycryface",
    r":\|": "mutedface",
    r":d+": "laughface",
    r"<3": "loveicon",
    r"\.{2,}": "threedot",
    r"-{1,}>{1,}": "arrow",
    r"={1,}>{1,}": "arrow",
    r"(\d+)h": r"\1 giờ",
    r"(\d+)'": r"\1 phút",
    r"(\d+)trieu": r"\1 triệu",
    r"(\d+)\s?tr": r"\1 triệu",
    r"blut\w+": "bluetooth",
}

MISPELLED_WORD_REPLACE = {
    "taay": "tây",
    "gđ": "gia đình",
    "nch": "nói chuyện",
    "dới": "với",
    "quéo": "quá",
    "thgian": "thời gian",
    "trc": "trước",
    "gank": "gánh",
    "gánh tem": "gánh team",
    "cty": "công ty",
    "film": "phim",
    "chếc": "chết",
    "chec": "chết",
    "hum": "hôm",
    "mẹt": "mặt",
    "bth": "bình thường",
    "bthg": "bình thường",
    "vde": "vấn đề",
    "lg.bt": "lgbt",
    "eo gi bi ti": "lgbt",
    "ốt ca": "oscar",
    "bik": "biết",
    "thik": "thích",
    "ytb": "youtube",
    "fuk cuốc": "phục quốc",
    "vs": "với",
    "cj": "chị",
    "chs": "chơi",
    "trg": "trong",
    "gr": "group",
    "zời": "trời",
    "tuỵt": "tuyệt",
    "tuỵt zời": "tuyệt vời",
    "jztr": "gì vậy trời",
    "dl": "deadline",
    "nhe": "nha",
    "cà-phê": "cà phê",
    "cafe": "cà phê",
    "oach": "oách",
    "ck": "chồng",
    "vk": "vợ",
    "cức": "cứt",
    "mún": "muốn",
    "iu": "yêu",
    "lũi": "lỗi",
    "gei": "gay",
    "gkke": "ghê",
    "ghia": "ghê",
    "mn": "mọi người",
    "mng": "mọi người",
    "nhìu": "nhiều",
    "nx": "nữa",
    "cx": "cũng",
    "del": "đéo",
    "đếu": "đéo",
    "dell": "đéo",
    "đé0": "đéo",
    "mẽo": "mỹ",
    "mĩ": "mỹ",
    "ngoo": "ngu",
    "3///": "3 que",
    "ba ///": "3 que",
    "///": "3 que",
    "ba que": "3 que",
    "dlv": "dư luận viên",
    "dảk": "dark",
    "xaolol": "xạo lồn",
    "xao lol": "xạo lồn",
    "cg": "cũng",
    "oce": "ok",
    "okla": "ok",
    "oke": "ok",
    "okay": "ok",
    "oki": "ok",
    "lu ân": "luân",
    "lun": "luôn",
    "luon": "luôn",
    "luông": "luôn",
    "ya": "yeah",
    "yah": "yeah",
    "yeh": "yeah",
    "ye": "yeah",
    "cã": "cả",
    "thiệc": "thật",
    "thiệt": "thật",
    "nka": "nha",
    "hẻ": "hả",
    "ik": "đi",
    "mik": "mình",
    "j": "gì",
    "jz": "gì vậy",
    "sô bít": "showbiz",
    "sâu bít": "showbiz",
    "uh": "ừ",
    "ovtk": "overthinking",
    "duocj": "được",
    "dc": "được",
    "đc": "được",
    "roài": "rồi",
    "vid": "video",
    "z": "vậy",
    "zậy": "vậy",
    "zị": "vậy",
    "zui": "vui",
    "zẻ": "vẻ",
    "zịt tân": "việt tân",
    "dui dẻ": "vui vẻ",
    "trùi": "trời",
    "rùi": "rồi",
    "dth": "dễ thương",
    "thw": "thương",
    "hem": "không",
    "hok": "không",
    "ko": "không",
    "k": "không",
    "hok": "không",
    "hk": "không",
    "tứk": "tức",
    "tưk": "tức",
    "chếт": "chết",
    "đé0": "đéo",
    "buoм": "bướm",
    "lфи": "lồn",
    "lìn": "lồn",
    "Iồn": "lồn",
    "facebooj": "facebook",
    "fb": "facebook",
    "ins": "instagram",
    "i ponr": "iphone",
    "thi.ê.t <url>": "thiệt mạng",
    "cừ": "cười",
    "bịnh": "bệnh",
    "đin": "điên",
    "nнân": "nhân",
    "kộn": "lộn",
    "sảu": "sủa",
    "đ'": "đéo",
    "mèo đẹ": "đéo mẹ",
    "đhs": "đéo hiểu sao",
    "rân chủ": "dân chủ",
    "be chím": "chim bé",
    "sờ trim mơ": "streamer",
    "trim": "chim",
    "lòn": "lồn",
    "đũy": "đĩ",
    "đix": "đĩ",
    "đũy": "đĩ",
    "ear": "ỉa",
    "ẻ": "ỉa",
    "ỉe": "ỉa",
    "🆑": "cái lồn",
    "cht": "chết",
    "người rưng": "người dưng",
    "anh aays ddax bij ddwa ddeesn laau ddaif tinhf ais cura morderkaiser": "anh ấy đã bị đưa đến lâu đài tình ái của mordekaiser",
    "anh se dua\nanh se dua em di toi brazil 🗣️🗣️🇧🇷": "anh sẽ đưa\nanh sẽ đưa em đi tới brazil 🗣️🗣️🇧🇷",
    "slip": "sleep",
    "xợ": "sợ",
    "nu mước": "mu nước",
    "mòe": "mèo",
    "cl": "cái lồn",
    "tiktik": "tiktok",
    "tht": "thật",
    "adu": "á đù",
    "ukraine la con benh thi o vn co mot thang tuong benh hoan. ngao da vo hoc . ong bao thang he 43 tuoi lam tt con ong bn tuoi ma ong len lam he cho chung chuoi!": "ukraine là con bệnh thì ở việt nam thằng tướng bệnh hoạn. ngáo đá vô học . ông bảo thằng hề 43 tuổi làm tổng thống còn ông bao nhiêu tuổi mà ông lên làm hề cho chúng chửi",
    "vn": "việt nam",
    "tq": "trung quốc",
    "chưởi": "chửi",
    "admu": "admin",
    "ad": "admin",
    "adm": "admin",
    "ếu": "éo",
    "pà": "bà",
    "đựt": "được",
    "khum": "không",
    "hong": "không",
    "ms": "mới",
    "bh": "bây giờ",
    "qias": "quả",
    "zô": "vô",
    "dô": "vô",
    "cmt": "comment",
    "cc": "con cặc",
    "dm": "địt mẹ",
    "đm": "địt mẹ",
    "vc": "vãi cặc",
    "vcd": "vãi cả đái",
    "vcđ": "vãi cả đái",
    "cak": "cặc",
    "cặk": "cặc",
    "đcm": "địt con mẹ",
    "đkm": "địt con mẹ",
    "dcm": "địt con mẹ",
    "dkm": "địt con mẹ",
    "vcl": "vãi cả lồn",
    "vkl": "vãi cả lồn",
    "vl": "vãi lồn",
    "vler": "vãi lồn",
    "cmn": "con mẹ nó",
    "cmnr": "con mẹ nó rồi",
    "cmm": "con mẹ mày",
    "clm": "con lồn mẹ",
    "trdu": "trời đụ",
    "lùm mía": "lồn má",
    "đjt": "địt",
    "djt": "địt",
    "kac": "cặc",
    "lz": "lồn",
    "loz": "lồn",
    "lol": "lồn",
    "lồl": "lồn",
    "lul": "lồn",
    "lz": "lồn",
    "lone": "lồn",
    "vlz": "vãi lồn",
    "cũm": "cũng",
    "mịa": "mẹ",
    "mọe": "mẹ",
    "nghe muon ia": "nghe muốn ỉa",
    "l i k e": "like",
    "tag": "tag",
    "xoq": "xong",
    "xog": "xong",
    "triến sỹ": "chiến sĩ",
    "triếk sỹ": "chiến sĩ",
    "ukraina": "ukraine",
    "u cà": "ukraine",
    "mắ": "má",
    "móa": "má",
    "mé": "má",
    "douma": "đụ má",
    "duma": "đụ má",
    "qá": "quá",
    "oi": "ơi",
    "êi": "ơi",
    "rp": "report",
    "rì pọt": "report",
    "kh": "không",
    "đá¡i": "đái",
    "nc": "nước",
    "giúng": "giống",
    "ngta": "người ta",
    "chiện": "chuyện",
    "ún": "uống",
    "um xùm": "um sùm",
    "ae": "anh em",
    "md": "mất dạy",
    "hửm": "hả",
    "ròi": "rồi",
    "gòi": "rồi",
    "gồi": "rồi",
    "xoq": "xong",
    "xg": "xong",
    "xog": "xong",
    "gocu": "goku",
    "sơn gocu": "son goku",
    "cgi": "cái gì",
    "t.â.m t.h.ầ.n": "tâm thần",
    "gs.": "giáo sư",
    "ts.": "tiến sĩ",
    "iem": "em",
    "hỉu": "hiểu",
    "đb": "đầu buồi",
    "ngiu": "người yêu",
    "ny": "người yêu",
    "chì chít": "chí chiết",
    "chít": "chết",
    "hix": "hic",
    "đth": "điện thoại",
    "ụa": "ủa",
    "riu": "real",
    "riel": "real",
    "gv": "giáo viên",
    "katyusha": "kachiusa",
    "vjp": "vip",
    "bruh": "bro",
    "uoc": "ước",
    "kiu": "kêu",
    "hổ trợ": "hỗ trợ",
    "cdm": "cộng đồng mạng",
    "cđm": "cộng đồng mạng",
    "tbg": "trư bát giới",
    "pết": "page",
    "ó": "á",
    "chúnh": "chúng",
    "thoai": "thôi",
    "vch": "vãi chưởng",
    "qtam": "quan tâm",
    "cụa": "của",
    "mọincon": "mọi con",
    "xì trâu": "xì trây",
    "trung binh xi tray o moi truong tu nhien cua chung": "trung bình xì trây ở môi trường tự nhiên của chúng",
    "ma tóe": "ma túy",
    "ℬ𝒶̆́𝒸 𝓀𝓎̀": "bắc kỳ",
    "cái=": "cái bằng",
    "nốn lừng": "nứng lồn",
    "giếт": "giết",
}

uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"


def loaddicchar():
    dic = {}
    char1252 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        "|"
    )
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic


dicchar = loaddicchar()


def convert_unicode(txt):
    return re.sub(
        r"à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ",
        lambda x: dicchar[x.group()],
        txt,
    )


vowel_table = [
    ["a", "à", "á", "ả", "ã", "ạ", "a"],
    ["ă", "ằ", "ắ", "ẳ", "ẵ", "ặ", "aw"],
    ["â", "ầ", "ấ", "ẩ", "ẫ", "ậ", "aa"],
    ["e", "è", "é", "ẻ", "ẽ", "ẹ", "e"],
    ["ê", "ề", "ế", "ể", "ễ", "ệ", "ee"],
    ["i", "ì", "í", "ỉ", "ĩ", "ị", "i"],
    ["o", "ò", "ó", "ỏ", "õ", "ọ", "o"],
    ["ô", "ồ", "ố", "ổ", "ỗ", "ộ", "oo"],
    ["ơ", "ờ", "ớ", "ở", "ỡ", "ợ", "ow"],
    ["u", "ù", "ú", "ủ", "ũ", "ụ", "u"],
    ["ư", "ừ", "ứ", "ử", "ữ", "ự", "uw"],
    ["y", "ỳ", "ý", "ỷ", "ỹ", "ỵ", "y"],
]
keyboard_accents_table = ["", "f", "s", "r", "x", "j"]

nguyen_am_to_ids = {}

for i in range(len(vowel_table)):
    for j in range(len(vowel_table[i]) - 1):
        nguyen_am_to_ids[vowel_table[i][j]] = (i, j)


def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == "q":
                chars[index] = "u"
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == "g":
                chars[index] = "i"
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = vowel_table[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = vowel_table[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowel_table[x][dau_cau]
                else:
                    chars[1] = (
                        vowel_table[5][dau_cau]
                        if chars[1] == "i"
                        else vowel_table[9][dau_cau]
                    )
            return "".join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowel_table[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return "".join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = vowel_table[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = vowel_table[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = vowel_table[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return "".join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True


def normalize_vietnamese_accents(sentence):
    """
    Chuyển câu tiếng việt về chuẩn gõ dấu kiểu cũ.
    :param sentence:
    :return:
    """
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r"(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)", r"\1/\2/\3", word).split("/")
        # print(cw)
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])
        words[index] = "".join(cw)
    return " ".join(words)


def normalize(text: str, track_change=False):
    # Lowercasing
    text = text.lower()

    # NFKC
    text = unicodedata.normalize("NFKC", text)
    if track_change:
        print("NFKC: ", text)

    # Remove dup trailing chars (troiiiii -> troi)
    text = DUP_TRALLING_RE.sub(r"\1", text)
    if track_change:
        print("Dedup trailing: ", text)

    # Replace special symbol to word
    for pttn, repl in SPECIAL_WORD_SUBSTITUTION.items():
        text = re.sub(rf"{pttn}", f" {repl} ", text)
    if track_change:
        print("Replace special word: ", text)

    # text = text.replace("<username>", "@username")
    # Correct misspelled word
    # def replace(match):
    #     orig = match.group(0)
    #     word = " " + MISPELLED_WORD_REPLACE.get(orig, orig) + " "
    #     return word

    # text = SYLLABLE_RE.sub(replace, text)
    # if track_change:
    #     print("Correct misspelled word: ", text)

    # Normalize string encoding
    text = convert_unicode(text)
    if track_change:
        print("Normalize string encoding: ", text)

    # Vietnamese diacritic normalization
    text = normalize_vietnamese_accents(text)
    if track_change:
        print("Vietnamese unicode normalization: ", text)

    # Eliminate grouping separator (9.000 / 9,000 -> 9000)
    text = GROUPING_SEPARATOR_RE.sub("", text)
    if track_change:
        print("Eliminate decimal delimiter: ", text)

    # Split between value and unit (300km -> 300 km)
    text = VALUE_UNIT_RE.sub(r"\1 \2", text)
    if track_change:
        print("Split between value and unit: ", text)

    # Split by punctuations
    text = " ".join(filter(None, PUNCTUATION_RE.split(text)))
    if track_change:
        print("Split by punctuations: ", text)

    # Split by emojis
    # text = " ".join(EMOJI_RE.split(text))
    text = emoji.replace_emoji(text, lambda e, d: f" {e} ")
    if track_change:
        print("Split by emojis: ", text)

    # Word segmentation
    # text = " ".join(vncorenlp.word_segment(text))

    return text