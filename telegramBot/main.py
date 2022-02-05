import logging
from YOLO import OBJ_DETECTOR
from DEPTH_ESTIMATION import DEPTH_ESTIMATOR, preprocess_image, compose, read_classes, read_anchors, scale_boxes, get_colors_for_classes, draw_boxes
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import emoji
import math

from telegram import  InlineKeyboardButton, InlineKeyboardMarkup, Update, ForceReply,chataction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler


# Full Network(Object_detector+Depth-Estimator) function
def plotter(img, dataframe,id):
    df_full = DEPTH_ESTIMATOR(img, dataframe,id)
    return df_full

def drawwer(img, df, id):
    image, image_data = preprocess_image(img, model_image_size=(608, 608))
    class_names = read_classes("coco_classes.txt")
    colors = get_colors_for_classes(len(class_names))
    out_boxes = np.zeros((len(df.index), 4))
    out_classes = np.zeros((len(df.index)))
    out_scores = np.zeros((len(df.index)))
    DEPTHS=[]
    for index, row in df.iterrows():
        DEPTHS.append(row['depth'])

    for index, row in df.iterrows():
        out_boxes[index, :] = [row['ymin'], row['xmin'], row['ymax'], row['xmax']]
        out_classes[index] = row['class']
        out_scores[index] = row['confidence']

    o2 = tf.convert_to_tensor(out_boxes)
    o3 = tf.convert_to_tensor(out_classes, dtype=tf.int32)
    o5 = tf.convert_to_tensor(out_scores)
    draw_boxes(image, o2, o3, class_names, o5, DEPTHS)
    image.save(f'E:Test_fil{id}.jpg', quality=200)
    return df


# gif function!
def rnd_gif(rnd):
    if rnd ==1:
        name='Karim'
    elif rnd==2:
        name='Mahmood'
    elif rnd==3:
        name='Khers'
    else:
        name='Roshan'
    return name


def FILTER(df,state):
    df_filtered = df
    if state=='1':
        for index, row in df_filtered.iterrows():
            if row['depth']<2:
                df_filtered.drop([index],inplace = True)
        state_text='دورتر از 2 متر'

    elif state=='2':
        for index, row in df_filtered.iterrows():
            if row['depth']>2:
                df_filtered.drop([index],inplace = True)
        state_text = 'نزدیکتر از 2 متر'

    elif state=='3':
        for index, row in df_filtered.iterrows():
            if row['depth']<3:
                df_filtered.drop([index],inplace = True)
        state_text = 'دورتر از 3 متر'

    elif state =='4':
        for index, row in df_filtered.iterrows():
            if row['depth']>3:
                df_filtered.drop([index],inplace = True)
        state_text = 'نزدیکتر از 3 متر'

    elif state=='5':
        for index, row in df_filtered.iterrows():
            if row['depth']<4:
                df_filtered.drop([index],inplace = True)
        state_text = 'دورتر از 4 متر'

    elif state=='6':
        for index, row in df_filtered.iterrows():
            if row['depth']>4:
                df_filtered.drop([index],inplace = True)
        state_text = 'نزدیکتر از 4 متر'

    df_filtered.reset_index(inplace=True)
    return df_filtered, state_text

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


def button(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    query.answer()
    if query.data=='10':
        query.edit_message_text(text="خب حله! یه عکس دیگه بفرست! " + emoji.emojize(':v:', use_aliases=True))
    else:
        query.edit_message_text(text="خب بریم سراغ اینکه پیداشون کنیم!")
        df_full = pd.read_pickle(f'df_full{update.effective_user.id}')
        df_filtered, state_text = FILTER(df_full, query.data)
        df_filtered = drawwer(img=f'Test{update.effective_user.id}.jpg', df=df_filtered, id=update.effective_user.id)
        if df_filtered.shape[0]!=0:
            with open(f'E:Test_fil{update.effective_user.id}.jpg', 'rb') as photo:
                update.effective_message.reply_photo(photo,caption=f'اشیاء {state_text} رو مشخص کردم برات. برو حال کن! ' + emoji.emojize(':joy:', use_aliases=True))
                update.effective_message.reply_text(text='ما هستیم اینجا! میتونی بازم عکس جدید بفرستی! ' + emoji.emojize(':eyes:', use_aliases=True))
        else:
                update.effective_message.reply_text(text=f'والا شئ ای با عمق {state_text} پیدا نکردم واست تو عکس :(')




# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    logger.info("User %s started the conversation.", user.first_name)
    update.message.reply_markdown_v2(
        fr'سلام {user.mention_markdown_v2()}\! '
    )
    update.message.reply_markdown_v2(
        fr'ما اینجا قراره پروژه دیپ لرنینگمون رو تست کنیم\! '
    )
    update.message.reply_markdown_v2(
        'چطوره یه عکس برامون بفرستی؟\! \n توضیح: هر عکسی\! ولی ترجیحا از دنیای واقعی باشه  و اینکه اینجا همه چی privateعه و چیزی ذخیره نمیشه\! '
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('یه عکس بفرست! \n بقیه کار رو بسپار به ما')


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text('سلطان! عکس بفرست! \nنیومدیم که حرف بزنیمااا :))')


def receiver(update: Update, context: CallbackContext) -> None:
    """get a photo from user and save it"""
    newFile = update.message.effective_attachment[-1].get_file()
    newFile.download(f'Test{update.effective_user.id}.jpg')
    update.message.reply_text('خب یه چند لحظه وایسا لطفا!')
    update.message.reply_chat_action(action='upload_photo')
    df = OBJ_DETECTOR(img_path=f'Test{update.effective_user.id}.jpg')
    df_full = plotter(f'Test{update.effective_user.id}.jpg', df,update.effective_user.id)
    with open(f'E:Test{update.effective_user.id}.jpg', 'rb') as photo:
        update.message.reply_photo(photo)
    full_dict = df_full.to_dict(orient='dict')
    OBJECTS=[]
    CONF=[]
    DPT=[]
    for k in range(len(full_dict["name"])):
        OBJECTS.append(full_dict['name'][k])
        CONF.append(round(full_dict['confidence'][k],2))
        DPT.append(full_dict['depth'][k])
    if len(OBJECTS)!=0:
        TEXT='چیزایی که تو عکس پیدا کردیم:\n\n'
        for l in range(len(full_dict["name"])):
            TEXT = TEXT+ str(OBJECTS[l])+''+'    confidence: '+str(CONF[l])+'    depth= ' +str(DPT[l])+ 'm'+ '\n'
        TEXT = TEXT+'\n'+'برای اینکه شکل شلوغ نشه، مقادیر اطمینان هر باکس رو دیگه تو شکل نذاشتیم :))'
        update.message.reply_text(TEXT)
        #rnd = random.randint(1, 5)
        #with open(rnd_gif(rnd)+'.mp4', 'rb') as ani:
            #update.message.reply_animation(animation=ani)
        df.to_pickle(f'df_full{update.effective_user.id}')

        """Sends a message with 6 inline buttons attached."""
        keyboard = [
            [
                InlineKeyboardButton("دورتر از 2 متر", callback_data='1'),
                InlineKeyboardButton("نزدیکتر از 2 متر", callback_data='2'),
            ],
            [
                InlineKeyboardButton("دورتر از 3 متر", callback_data='3'),
                InlineKeyboardButton("نزدیکتر از 3 متر", callback_data='4'),
            ],
            [
                InlineKeyboardButton("دورتر از 4 متر", callback_data='5'),
                InlineKeyboardButton("نزدیکتر از 4 متر", callback_data='6'),
            ],
            [
                InlineKeyboardButton("ولش کن! بریم یه عکس دیگه! "+ emoji.emojize(':grimacing:', use_aliases=True), callback_data='10'),
            ],

        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        update.message.reply_text('اگه بخوای فقط اشیاء با عمق مشخصی رو تو عکس ببینی، کدوم اشیاء؟؟', reply_markup=reply_markup)
    else:
        TEXT = 'چیزی پیدا نکردم والا تو این عکس :(  یه عکس دیگه بفرست'
        update.message.reply_text(TEXT)


def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("5228411403:AAGOqjwHBb9qmEXCyLbFueORCjTYWa2bQ2k")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(MessageHandler(Filters.photo, receiver))
    updater.dispatcher.add_handler(CallbackQueryHandler(button))

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(~Filters.photo & ~Filters.command, echo))


    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()