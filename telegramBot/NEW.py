import logging
from YOLO import OBJ_DETECTOR
from DEPTH_ESTIMATION import DEPTH_ESTIMATOR
import random
import math

from telegram import Update, ForceReply,chataction
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


# Full Network(Object_detector+Depth-Estimator) function
def plotter(img, dataframe,id):
    df_full = DEPTH_ESTIMATOR(img, dataframe,id)
    return df_full

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

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
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
    rnd = random.randint(1, 5)
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
        with open(rnd_gif(rnd)+'.mp4', 'rb') as ani:
            update.message.reply_animation(animation=ani)
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

    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler((Filters.text | Filters.document) & ~Filters.command, echo))


    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()