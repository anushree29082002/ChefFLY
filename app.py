import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline, set_seed
from transformers import AutoTokenizer
import enum
import random
import time
from typing import List, Tuple
# from streamlit_searchbox import st_searchbox
# from streamlit_javascript import st_javascript
import requests

from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

from PIL import (
    ImageFont,
)

import os
import re
from examples import EXAMPLES
import dummy
import meta
from utils import ext
from utils.api import generate_cook_image
from utils.draw import generate_food_with_logo_image, generate_recipe_image
from utils.st import (
    remote_css,
    local_css,

)
from utils.utils import (
    load_image_from_url,
    load_image_from_local,
    image_to_base64,
    pure_comma_separation,
)

class TextGeneration:
    def __init__(self):
        self.debug = False
        self.dummy_outputs = dummy.recipes
        self.tokenizer = None
        self.generator = None
        self.api_ids = []
        self.api_keys = []
        self.api_test = 2
        self.task = "text2text-generation"
        self.model_name_or_path = "flax-community/t5-recipe-generation"
        self.color_frame = "#ffffff"
        self.main_frame = "asset/frame/recipe-bg-3.png"
        self.no_food = "asset/frame/no_food.png"
        self.no_food_2 = "asset/frame/no_food_2.png"
        self.logo_frame = "asset/images/Red and Blue Simple Personal Chef Logo (2).png"
        self.chef_frames = {
            "scheherazade": "asset/frame/food-image-logo-bg.png",
            "giovanni": "asset/frame/food-image-logo-bg.png",
        }
        self.fonts = {
            "title": ImageFont.truetype("asset/fonts/Poppins-Bold.ttf", 70),
            "sub_title": ImageFont.truetype("asset/fonts/Poppins-Medium.ttf", 30),
            "body_bold": ImageFont.truetype("asset/fonts/Montserrat-Bold.ttf", 22),
            "body": ImageFont.truetype("asset/fonts/Montserrat-Regular.ttf", 18),

        }
        set_seed(42)

    def _skip_special_tokens_and_prettify(self, text):
        recipe_maps = {"<sep>": "--", "<section>": "\n"}
        recipe_map_pattern = "|".join(map(re.escape, recipe_maps.keys()))

        text = re.sub(
            recipe_map_pattern,
            lambda m: recipe_maps[m.group()],
            re.sub("|".join(self.tokenizer.all_special_tokens), "", text)
        )

        data = {"title": "", "ingredients": [], "directions": []}
        for section in text.split("\n"):
            section = section.strip()
            if section.startswith("title:"):
                data["title"] = " ".join(
                    [w.strip().capitalize() for w in section.replace("title:", "").strip().split() if w.strip()]
                )
            elif section.startswith("ingredients:"):
                data["ingredients"] = [s.strip() for s in section.replace("ingredients:", "").split('--')]
            elif section.startswith("directions:"):
                data["directions"] = [s.strip() for s in section.replace("directions:", "").split('--')]
            else:
                pass

        return data

    def load_pipeline(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.generator = pipeline(self.task, model=self.model_name_or_path, tokenizer=self.model_name_or_path)

    def load_api(self):
        app_ids = os.getenv("EDAMAM_APP_ID")
        app_ids = app_ids.split(",") if app_ids else []
        app_keys = os.getenv("EDAMAM_APP_KEY")
        app_keys = app_keys.split(",") if app_keys else []

        if len(app_ids) != len(app_keys):
            self.api_ids = []
            self.api_keys = []

        self.api_ids = app_ids
        self.api_keys = app_keys

    def load(self):
        self.load_api()
        if not self.debug:
            self.load_pipeline()

    def prepare_frame(self, recipe, chef_name):
        frame_path = self.chef_frames[chef_name.lower()]
        food_logo = generate_food_with_logo_image(frame_path, self.logo_frame, recipe["image"])
        frame = generate_recipe_image(
            recipe,
            self.main_frame,
            food_logo,
            self.fonts,
            bg_color="#ffffff"
        )
        return frame

    def generate(self, items, generation_kwargs):
        recipe = self.dummy_outputs[0]
        if not self.debug:
            generation_kwargs["num_return_sequences"] = 1
            generation_kwargs["return_tensors"] = True
            generation_kwargs["return_text"] = False
            generated_ids = self.generator(
                items,
                **generation_kwargs,
            )[0]["generated_token_ids"]
            recipe = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            recipe = self._skip_special_tokens_and_prettify(recipe)

        if self.api_ids and self.api_keys and len(self.api_ids) == len(self.api_keys):
            test = 0
            for i in range(len(self.api_keys)):
                if test > self.api_test:
                    recipe["image"] = None
                    break
                image = generate_cook_image(recipe["title"].lower(), self.api_ids[i], self.api_keys[i])
                test += 1
                if image:
                    recipe["image"] = image
                    break
        else:
            recipe["image"] = None

        return recipe

    def generate_frame(self, recipe, chef_name):
        return self.prepare_frame(recipe, chef_name)
    
@st.cache(allow_output_mutation=True)
def load_text_generator():
    generator = TextGeneration()
    generator.load()
    return generator


chef_top = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95,
    "num_return_sequences": 1
}
chef_beam = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "num_beams": 5,
    "length_penalty": 1.5,
    "num_return_sequences": 1
}


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
"""

def main():

    st.set_page_config(
        page_title="ChefFLy",
        page_icon="🍲",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(hide_menu_style, unsafe_allow_html=True)

    generator = load_text_generator()
    remote_css("https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&family=Poppins:wght@600&display=swap")
    local_css("asset\css\style.css")
    col1, col2 = st.columns([6,4])
    with col2:
        st.markdown("""
            <style>
                .fullScreenFrame > div {
                    display: flex;
                    justify-content: center;
                }
            </style>
        """,unsafe_allow_html=True)
        st.image(load_image_from_local("asset\images\Red and Blue Simple Personal Chef Logo (2).png"), width=425)
        st.markdown(meta.SIDEBAR_INFO, unsafe_allow_html=True)
        

        with st.expander("Where did this story start?", expanded=True):
            st.markdown(meta.STORY, unsafe_allow_html=True)
            st.image(load_image_from_local("asset\images\Capture.png"), width=475)
    with col1:
        st.markdown(meta.HEADER_INFO, unsafe_allow_html=True)

        st.markdown(meta.CHEF_INFO, unsafe_allow_html=True)

        selectbox_css = """ 
            # <link rel="preconnect" href="https://fonts.googleapis.com">
            # <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            # <link href="https://fonts.googleapis.com/css2?family=Amita:wght@400;700&display=swap" rel="stylesheet">
            <style>
            .st-dg {
                padding-right: 0.5rem;
                font-size: 19px;
                font-family: 'Poppins';
            }

            .css-l3i8zm {
                font-size: 23px;
                color: rgb(0, 0, 0);
                display: flex;
                visibility: visible;
                margin-bottom: 0.5rem;
                height: auto;
                min-height: 1.5rem;
                vertical-align: middle;
                flex-direction: row;
                -webkit-box-align: center;
                align-items: center;
                font-weight: bold;
                font-family: 'Poppins'
            }

            .css-184tjsw p {
                word-break: break-word;
                font-size: 20px;
                font-weight: bold;
            }

            .css-18e3th9 {
                flex: 1 1 0%;
                width: 100%;
                padding-top:1%;
                min-width: auto;
                max-width: initial;
            }
            .css-9ycgxx {
                margin-bottom: 0.25rem;
                font-size: 20px;
                font-weight: bold;
                font-family: 'Poppins';
            }

            css-kxx3wx {
                display: inline-flex;
                -webkit-box-align: center;
                align-items: center;
                -webkit-box-pack: center;
                justify-content: center;
                font-weight: bolder;
                padding: 0.25rem 0.75rem;
                border-radius: 0.25rem;
                margin: 10px;
                line-height: 1.6;
                color: inherit;
                width: auto;
                user-select: none;
                background-color: rgb(219, 225, 223);
                border: 1px solid rgba(0, 0, 0, 0.2);
            }

            .css-1t8kd5l {
                color: rgba(0, 0, 0, 0.6);
                font-size: 18px;
                line-height: 1.25;
            }

            button, input, optgroup, select, textarea {
                font-size: 20px;
            }

            p, ol, ul, dl {
                margin: 0px 0px 1rem;
                padding: 0px;
                font-size: 1rem;
                font-weight: bold;
                font-size: 20px;
                font-family: 'Poppins';
            }

            .css-aoxipv p {
                word-break: break-word;
                font-size: 15px;
                font-weight:bold;
            }

            .st-g1 {
                color: rgba(0, 0, 0, 0.6);
                font-size: 20px;
            }

            .st-er {
                min-height: 15px;
                font-size: 20px;
                font-weight: bold;
                font-family: 'Poppins';
            }

            .css-2arey {
                display: inline-flex;
                -webkit-box-align: center;
                align-items: center;
                -webkit-box-pack: center;
                justify-content: center;
                padding: 0.25rem 0.75rem;
                border-radius: 0.25rem;
                margin: 0px;
                line-height: 1.6;
                color: inherit;
                width: auto;
                user-select: none;
                background-color: #87A7B3;
                border: 1px solid rgba(0, 0, 0, 0.2);
                font-weight:bold;
                font-size:22px;

            }

            .css-1sslkv8 {
                font-family: "Source Sans Pro", sans-serif;
                margin-bottom: -1rem;
                color: #000000;
            }

            .contributors a.contributor {
                text-decoration: none;
                color: #1424c0;
                font-size: 20px;
            }

            .story-box p {
                font-size: 20px;
            }

            .css-aoxipv p {
                word-break: break-word;
                font-size: 20px;
            }
            .css-8ojfln {
                display: table-cell;
                font-size: 20px;
                font-weight: bold;
            }

            .css-10trblm {
                position: relative;
                flex: 1 1 0%;
                margin-left: calc(3rem);
                font-family: 'Amita', cursive;
                font-size: 60px;
                text-align: center;
                color: #00235b;
            }
            h1 {
                font-family: "Source Sans Pro", sans-serif;
                font-weight: 700;
                color: rgb(0, 0, 0);
                padding: 5px 0px 0px 0px;
                margin: 0px;
                line-height: 1.2;
            }

            .st-ae {
                font-family: "Poppins", sans-serif;
            }

            .css-l3i8zm {
                font-size: 20px;
                color: rgb(0, 0, 0);
                display: flex;
                visibility: visible;
                margin-bottom: 0.5rem;
                height: auto;
                min-height: 1.5rem;
                vertical-align: middle;
                flex-direction: row;
                -webkit-box-align: center;
                align-items: center;
                font-weight: bold;
            } 

            .css-1djdyxw {
                vertical-align: middle;
                display: flex;
                flex-direction: row;
                -webkit-box-align: center;
                align-items: center;
                font-size: 21px;
                font-family: 'Poppins';
            }         

            
            element.style {
                font-size: 20px;
            }

            .extra-info {
                font-weight: bold;
                font-family: 'Poppins',sans-serif;
            }

            .css-1v0mbdj {
                display: flex;
                flex-direction: column;
                -webkit-box-align: stretch;
                align-items: stretch;
                width: auto;
                -webkit-box-flex: 0;
                flex-grow: 0;
                margin-left: 20px;
            }

            
            element.style {
                width: 310px;
            }

            .css-k1ih3n {
                width: 100%;
                min-width: auto;
                max-width: initial;
                padding-top:1%;
            }

            .extra-info {
                font-weight: bold;
                font-family: 'Poppins',sans-serif;
                color: #0a0968;
            }
            </style>

            
        """

        st.markdown(selectbox_css,unsafe_allow_html=True)

        chef = st.selectbox("Choose your chef", index=0, options=["Chef Scheherazade", "Chef Giovanni"])

            
        prompts = list(EXAMPLES.keys()) + ["Custom"]
        prompt = st.selectbox(
            'Examples (select from this list)',
            prompts,
            index=0
        )

        # if prompt == "Custom":
        #     prompt_box = ""
        # else:
        #     prompt_box = EXAMPLES[prompt]
        
        global ingredients
        ingredients = [
        'apple','acidulated water','ackee','acorn squash','aduki beans','advocaat','agar-agar','ale','alfalfa sprouts','allspice','almond','almond essence','almond extract','amaranth','amaretti','anchovies','anchovy essence','angelica','angostura bitters','anise','apricot','apricot jam','arborio rice','arbroath smokie','argan oil','arrowroot','artichoke','asafoetida','asparagus','aubergine','avocado','bacon','bagel','baguette','baked beans','baking','baking powder','balsamic vinegar','bamboo shoots','banana','banana bread','barbary duck','barbecue sauce','barley','basil','basmati rice','bay boletes','bay leaf','beans','beansprouts','bechamel sauce','beef','beef consomme','beef dripping','beef mince','beef ribs','beef rump','beef sausage','beef stock','beef tomato','beer','beetroot','berry','betel leaves','beurre manie','bicarbonate of soda','bilberries','bird\'s-eye chillies','biscotti','biscuits','blachan','black beans','black bream','black eyed beans','black pepper','black pudding','black sesame seeds','black treacle','blackbean sauce','blackberry','blackcurrant','blackcurrant juice drink','blini','blood orange','blueberry','boar','bok choi','bonito','borage','borlotti beans','bouquet garni','braising steak','bramley apple','bran','brandy','brandy butter','brandy snaps','bratwurst','brazil nut','bread','bread roll','bread sauce','breadcrumbs','breadfruit','breadsticks','bresaola','brie','brill','brioche','brisket','broad beans','broccol','brot','brown brea','brown lenti','brown ric','brown sauc','brown shrim','brown suga','brussels sprout','buckwheat','buckwheat flour','bulgur wheat','bun','burger','butter','butter bean','buttercream icing','butterhead lettuce','buttermill','butternut squash','cabbage','caerphilly','cake','calasparra rice','calvados','camembert','campagne loaf','candied peel','cannellini beans','cape gooseberries','capers','capsicum','caramel','caraway seeds','cardamom','carob','carrageen moss','carrageen_moss','carrot','cashew','cassava','caster sugar','catfish','caul fat','cauliflower','cava','caviar','cavolo nero','cayenne pepper','celeriac','celery','celery seeds','champ','champagne','chanterelle mushrooms','chantilly cream','chapati flour','chapatis','charcuterie','chard','charlotte potato','chayote','cheddar','cheese','cheese sauce','cherry','cherry brandy','cherry tomatoes','chervil','cheshire','chestnut','chestnut mushrooms','chicken','chicken breast','chicken casserole','chicken leg','chicken liver','chicken soup','chicken stock','chicken thigh','chicken wing','chickpea','chickpea flour','chicory','chilli','chilli con carne','chilli oil','chilli paste','hilli powder','chilli sauce','chinese cabbage','chinese mushrooms','chinese pancake','chipotle','chips','chives','chocolate','chocolate biscuit','chocolate brownies','chocolate cake','chocolate mousse','chocolate truffle','chopped tomatoes','chorizo','choux pastry','christmas cake','christmas pudding','chuck and blade','chump','chutney','ciabatta','cider','cinnamon','citrus fruit','clams','clarified butter','clementine','clotted cream','cloves','cobnut','cockles','cocktail','cocoa butter','cocoa powder','coconut','coconut cream','coconut milk','coconut oil','cod','cod roe','coffee','coffee beans','coffee essence','coffee granules','coffee liqueur','cognac','cola','coleslaw','coley','collar','compote','comte','condensed milk','confectionery','coriander','coriander cress','coriander seeds','corn oil','corn syrup','corned beef','cornflour','cos lettuce','cottage cheese','coulis','courgette','court bouillon','couscous','crab','crab apple','crackers','cranberry','cranberry juice','cranberry sauce','crayfish','cream','cream cheese','cream liqueur','cream of tartar','cream soda','creamed coconut','creme fraiche','crepe','cress','crispbread','crisps','croissant','crostini','croutons','crudites','crumble','crystallised ginger','cucumber','cumberland sauce','cumin','curacao','curd','curd cheese','curly kale','currant bread','currants','curry','curry leaves','curry paste','curry powder','custard','custard powder','cuttlefish',
        'dab','daikon','damsons','dandelion','danish blue','dark chocolate','date','demerara sugar','demi-glace sauce','desiccated coconut','desiree','digestive biscuit','dijon mustard','dill','dim sum wrappers','dolcelatte','double cream','double gloucester','dover sole','dragon fruit','dried apricots','dried cherries','dried chilli','dried fruit','dried mixed fruit','dry sherry','duck','duck confit','duck fat','dulce de leche','dumplings','duxelles','edam','eel','egg','egg wash','egg white','egg yolk','elderberries','elderflower','emmental','english muffin','english mustard','escalope','evaporated milk','exotic fruit',
        'farfalle','fat','fennel','fennel seeds','fenugreek','feta','fettuccine','field mushroom','fig','fillet of beef','filo pastry','fish','fish roe','fish sauce','fish soup','five-spice powder','flageolet beans','flaked almonds','flank','flapjacks','flatbread','flatfish','fleur de sel','flour','flour tortilla','floury potato','flying fish','focaccia','foie gras','fondant icing','fondant potatoes','fontina cheese','food colouring','forced rhubarb','fortified wine','fragrant rice','frangipane','frankfurter','freekeh','french beans','french bread','french dressing','fresh coriander','fresh tuna','fromage frais','fruit','fruit brandy','fruit cake','fruit juice','fruit salad','fudge','fusilli','galangal','game','gammon','garam masala','garlic','garlic and herb cream cheese','garlic bread','gelatine','ghee','gherkin','giblets','gin','ginger','ginger ale','ginger beer','ginger biscuit','gingerbread','glace cherries','globe artichoke','glucose','gnocchi','goats\' cheese','goats\' milk','golden syrup','goose','goose fat','gooseberry','gorgonzola','gouda','grain','grape juice','grapefruit','grapefruit juice','grapes','grapeseed oil','gratin','gravy','gravy browning','green banana','green beans','green cabbage','green lentil','green tea','greengages','grey mullet','ground almonds','ground ginger','grouse','gruyere','guacamole','guava','guinea fowl','gurnard',
        'habanero chillies','haddock','haggis','hake','halibut','halloumi','ham','hare','haricot beans','harissa','hazelnut','azelnut oil','heart','herbal liqueur','herbal tea','herbes de provence','herbs','herring','hogget','hoisin sauce','hoki','hollandaise sauce','hominy','honey','honeycomb','horseradish','horseradish sauce','hot cross buns','hummus','hunza apricots','ice cream','iceberg lettuce','icing','icing sugar','irish stout','jaggery','jam','january king cabbage','japanese pumpkin','jelly','jerk seasoning','jersey royal potatoes','jerusalem artichoke','john dory','jujube','juniper berries','jus','kabana','kale','ketchup','ketjap manis','kidney','kidney beans','king edward','kipper','kirsch','kiwi fruit','kohlrabi','kumquat','lager''lamb','lamb breast','lamb chop','lamb fillet','lamb kidney','lamb loin','lamb mince','lamb neck','lamb rump','lamb shank','lamb shoulder','lamb stock','lancashire','langoustine','lard','lardons','lasagne','lasagne sheets','laverbread','leek','leftover turkey','leg of lamb','lemon','lemon balm','lemon curd','lemon juice','lemon sole','lemonade','lemongrass','lentils','lettuce','lime','lime cordial','lime juice','lime leaves','lime pickle','ling','lingonberry','linguine','liqueur','liquorice','little gem lettuce','liver','loaf cake','lobster','loganberry','long-grain rice','lovage','lychee','macadamia''macaroni','macaroon','mace','mackerel','madeira','madeira cake','madeleines','maize','malted grain bread','manchego','mandarin','mangetout','mango','mango chutney','mango juice','mango pickle','mangosteen','maple syrup','margarine','marjoram','marmalade','marrow','marrowfat peas','marsala wine','marshmallow','marzipan','mascarpone','mashed potato','matzo','mayonnaise','meat','medlars','megrim','melon','melon seeds','meringue','mesclun','milk','milk chocolate','milkshake','millet','millet flour','mince','mince pies','mincemeat','mint','mint sauce','mirepoix','mirin','miso','mixed berries','mixed dried beans','mixed fish','mixed nuts','mixed spice','mixed spices','molasses','monk\'s beard','monkfish','morel','mortadella','mozzarella','muesli','muffins','mulberries','mulled wine','mung beans','mushroom','mussels','mustard','mustard cress','mustard leaves','mustard oil','mustard powder','mustard seeds','mutton','naan bread''nachos','nashi','nasturtium','nectarine','nettle','new potatoes','nibbed almonds','noodle soup','noodles','nori','nougat','nut','nutmeg','oatcakes','oatmeal','oats','octopus','offal','oil','oily fish','okra','olive','olive oil','onion','orange','orange juice','orange liqueur','oregano','ouzo','oxtail','oyster','oyster mushrooms','oyster sauce','paella','pak choi','palm sugar','pancakes','pancetta','pandan leaves','paneer','panettone','papaya','pappardelle','paprika','parfait','parmesan','parsley','parsnip','partridge','passata','passion fruit','passion fruit juice','pasta','pastrami','pastry','pasty','pate','paw-paw','pea shoots','peach','peanut butter','peanut oil','peanuts','pear','pearl barley','peas','pecan','pecorino','pectin','peel','penne','pepper','peppercorn','pepperoni','perch','perry','pesto','pheasant','piccalilli','pickle','pickled onion','pie','pig cheeks','pigeon','pigeon peas','pike','pine nut','pineapple','pineapple juice','pink fir apple','pink peppercorn','pinto beans','piri-piri','pistachio','pitta bread','pizza','pizza base','plaice','plain flour','plantain','plum','polenta','pollack','pollock','pomegranate','pomegranate juice','pomelo','popcorn','poppy seeds','porcini','pork','pork belly','pork chop','pork fillet','pork leg','pork loin','pork mince','pork sausages','pork shoulder','pork spare rib','port','portobello mushrooms','potato','potato rosti','potato wedges','poultry','poussin','praline','prawn','prawn crackers','preserved lemons','preserves','prosciutto','prune','prune juice','pudding rice','puff pastry','pulled pork','pumpernickel bread','pumpkin','pumpkin seed','purple sprouting broccoli','puy lentils','quail','quail\'s egg','quatre-epices','quince','quinoa','rabbit','rack of lamb','radicchio','radish','rainbow chard','rainbow trout','raisins','raita','rapeseed oil','ras-el-hanout','raspberry','raspberry jam','ratafia biscuits','ratatouille','red cabbage','red leicester','red lentil','red mullet','red onion','red rice','red snapper','red wine','red wine vinegar','redcurrant','redcurrant jelly','rennet','rhubarb','rib of beef','rice','rice flour','rice noodles','rice pudding','rice vinegar','rice wine','ricotta','rigatoni','risotto','risotto rice','roast beef','roast chicken','roast lamb','roast pork','roast potatoes','roast turkey','roasted vegetables','rock salmon','rock salt','rocket','root beer','root vegetable','roquefort','rose wine','rosehip syrup','rosemary','rosewater','rouille','royal icing','rum','rump','unner beans','rye bread','rye flour','safflower oil''saffron','sage','salad','salad cream','salad leaves','salami','salmon','salsa','salsify','salt','salt beef','salt cod','sambuca','samphire','sardine','sashimi','satsuma','sauces','saucisson','sausage','savory','savoy cabbage','scallop','scotch bonnet chilli','scrag','sea bass','sea bream','sea salt','sea trout','seafood','seasoning','seaweed','seeds','self-raising flour','semolina','serrano ham','sesame oil','sesame seeds','seville orange','shallot','sharon fruit','shellfish','sherry','sherry vinegar','shiitake mushroom','shin','shortbread','shortcrust pastry','sichuan pepper','silverside','single cream','sirloin','skate','sloe','sloe gin','smoked cheese','smoked fish','smoked haddock','smoked mackerel','smoked salmon','smoked trout','snapper','soba noodles','soda','soda bread','sole','sorbet','sorrel','soup','sourdough bread','soured cream','soy sauce','soya beans','soya flour','soya milk','soya oil','spaghetti','spaghetti squash','sparkling wine','spelt','spelt flour','spices','spinach','split peas','sponge cake','spring greens','spring onion','spring roll wrappers','squash','squid','star anise','starfruit','steak','stem ginger','stew','stewing lamb','stilton','stock','straw mushroom','strawberry','strawberry jam','strega liqueur','strong white flour','stuffing','sucralose','suet','sugar','sugar-snap peas','sultanas','sumac','summer cabbage','summer fruit','sunflower oil','sunflower seed','sushi rice','swede','sweet potato','sweet sherry','sweetbread','sweetcorn','swiss chard','swiss rolls and roulades','swordfish','syrup','t-bone steak''tabasco','taco','tagliatelle','tahini','taleggio','tamari','tamarillo','tamarind','tangerine','tapenade','tapioca','taro','tarragon','tartare sauce','tayberry','tea','tempura','tequila','teriyaki','teriyaki sauce','terrine','thai basil','thyme','tilapia','tinned tuna','toffee','tofu','tomatillo','tomato','tomato chutney','tomato juice','tomato puree','tonic','topside','tortellini','tripe','trout','truffle','truffle oil','turbot','turkey','turkey breast','turkey mince','turkish delight','turmeric','turnip','unleavened bread','vacherin','vanilla essence','vanilla extract','vanilla pod','veal','vegetable oil','vegetable shortening','vegetable stock','vegetables','vegetarian sausage','venison','verjus','vermicelli','vermouth','vine leaves','vinegar','vodka','vodka cocktail','waffles''walnut','walnut oil','wasabi','water chestnut','watercress','watermelon','waxy potato','webbs lettuce','wensleydale','wheatgerm','whelk','whipping cream','whisky','whisky cocktail','whisky liqueur','white bread','white cabbage','white chocolate','white fish','white pepper','white wine','white wine vinegar','whitebait','whitecurrant','whiting','whole wheat pasta','wholegrain mustard','wholemeal bread','wholemeal flour','wild duck','wild garlic','wild mushrooms','wild rice','wine','winkles','wood pigeon','worcestershire sauce','wraps','yam','yeast','yellow lentil','yoghurt','zander','zest','Coriander seeds','Aniseed','carom seeds','onion seeds','Licorice',
        
        'sabudana','rava','pohe','lal mirchi masala','biryani masala','chicken masala','mutton masala','pudina','laung','kali mirch','choti elaichi','badi elaichi','tej patta','sukhi lal mirch','Javitri','pathar phool','saunf','ajwain','kalonji','rai','hing','dhania powder','aamchur','kadhipatta','dhania','ghee','madh','mulethi','batata','shimla mirchi','vanga','bengan','urad dal','moong dal','toor dal','masur dal','harbara dal','kobi','flower','ghewada','methi','palak','shepu','chavli','shevgachya shenga','lal bhopla','dhudi bhopla','dodke','matar','gajar','kakdi','lasun','ala','limbu','beet','pavta','rajma','chole','haldi','imli','shenda namak','khaskas','til','khobra','shengdane','kokam','kaju','ratale','dahi','ice','french beans','kanda'
        ]

        if prompt == "Custom":
            items = st.multiselect("Select ingredients", ingredients)
            items = ", ".join(items)
            st.write('Selected ingredients: ', items)

        else:
            items = EXAMPLES[prompt]
            st.write("Selected ingredients: ", items)

        model = load_model('FV.h5')
        labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

        
        ingredients2 = ['Apple','Banana','Beetroot','Bell pepper','Cabbage','Capsicum','Carrot','Cauliflower','Chilli pepper','Corn','Cucumber','Eggplant','Garlic','Ginger','Grapes','Jalapeno','Kiwi','Lemon','Lettuce','Mango','Onion','Orange','Paprika','Pear','Peas','Pineapple','Pomegranate','Potato','Raddish','Soy beans','Spinach','Sweetcorn','Sweetpotato','Tomato','Turnip','Watermelon']
        
        def processed_img(img_path):
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            if predictions[0][predicted_class] < 0.5:
                return 'Cannot predict image'
            else:
                predicted_class_name = ingredients2[predicted_class]
                return predicted_class_name.capitalize()

        img_files = st.file_uploader("Choose Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

        predicted_ingredients = ''
        if img_files is not None:
            results = []
            for i, img_file in enumerate(img_files):
                img = Image.open(img_file).resize((250, 250))
                st.image(img, use_column_width=False)
                img_array = np.array(img)
                predicted_class = processed_img(img_file)
                if predicted_class == 'Cannot predict image':
                    st.write(f'Cannot Predict Image {i+1}')
                else:
                    results.append(predicted_class)
                    st.write(f"Image {i+1}: {predicted_class}")
            if len(results) > 0:
                predicted_ingredients = ", ".join(results)
        
        minor_ingredient_used = ''
        st.write('Minor ingredients required (Please do not enter minor ingredients in the text box above): ')
        if prompt == "Custom":
            minor_ingredient1 = st.checkbox('Oil', value=True)
            minor_ingredient2 = st.checkbox('Salt', value=True)
            minor_ingredient3 = st.checkbox('Red Chilli Powder', value=True)
            minor_ingredient4 = st.checkbox('Cumin Seeds', value=True)
            minor_ingredient5 = st.checkbox('Mustard Seeds', value=True)
            minor_ingredient6 = st.checkbox('Turmeric', value=True)

        else:
            minor_ingredient1 = st.checkbox('Oil', value=False)
            minor_ingredient2 = st.checkbox('Salt', value=False)
            minor_ingredient3 = st.checkbox('Red Chilli Powder', value=False)
            minor_ingredient4 = st.checkbox('Cumin Seeds', value=False)
            minor_ingredient5 = st.checkbox('Mustard Seeds', value=False)
            minor_ingredient6 = st.checkbox('Turmeric', value=False)

        if prompt == "Custom" and predicted_ingredients != '':
            ingredients_list = items + ', ' + predicted_ingredients
        elif prompt == "Custom" and predicted_ingredients == '':
            ingredients_list = items
        else:
            ingredients_list = EXAMPLES[prompt]
            
        minor_ingredient_used = ""

        if 'Oil' in items and minor_ingredient1:
            if ',oil' in minor_ingredient_used:
                minor_ingredient_used = ''

        if 'oil' not in items and minor_ingredient1:
            if ",oil" not in minor_ingredient_used:
                minor_ingredient_used += ", Oil"
        
        if 'salt' in items and minor_ingredient2:
            if ',salt' in minor_ingredient_used:
                minor_ingredient_used = ''
        
        if 'salt' not in items and minor_ingredient2:
            if 'salt' not in minor_ingredient_used:
                minor_ingredient_used += ', Salt'
        
        if 'red chilli powder' in items and minor_ingredient3:
            if ',red chilli powder' in minor_ingredient_used:
                minor_ingredient_used = ''
        
        if 'red chilli powder'not in items and minor_ingredient3:
            if ',red chilli powder' not in minor_ingredient_used:
                minor_ingredient_used += ', red chilli powder'
        
        if 'cumin seeds' in items and minor_ingredient4:
            if ',cumin seeds' in minor_ingredient_used:
                minor_ingredient_used = ''
        
        if 'cumin seeds' not in items and minor_ingredient4:
            if ',cumin seeds' not in minor_ingredient_used:
                minor_ingredient_used += ', cumin seeds'
        
        if 'mustard seeds' in items and minor_ingredient5:
            if ',mustard seeds' in minor_ingredient_used:
                minor_ingredient_used = ''
        
        if 'mustard seeds' not in items and minor_ingredient5:
            if ',mustard seeds' not in minor_ingredient_used:
                minor_ingredient_used += ', mustard seeds'
        
        if 'turmeric' in items and minor_ingredient6:
            if ',turmeric' in minor_ingredient_used:
                minor_ingredient_used = ''
        
        if 'turmeric' not in items and minor_ingredient6:
            if ',turmeric' not in minor_ingredient_used:
                minor_ingredient_used += ', turmeric'
        
        ingredients_list += minor_ingredient_used

        entered_items = st.empty()


    recipe_button = st.button('Get Recipe!')

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )
    if recipe_button:
        entered_items.markdown("**Generate recipe for:** " + ingredients_list )
        with st.spinner("Generating recipe..."):

            if not isinstance(items, str) or not len(items) > 1:
                entered_items.markdown(
                    f"**{chef}** would like to know what ingredients do you like to use in "
                    f"your food? "
                )
            else:
                gen_kw = chef_top if chef == "Chef Scheherazade" else chef_beam
                generated_recipe = generator.generate(ingredients_list, gen_kw)

                title = generated_recipe["title"]
                food_image = generated_recipe["image"]
                food_image = load_image_from_url(food_image, rgba_mode=True, default_image=generator.no_food)
                food_image = image_to_base64(food_image)

                ingredients = ext.ingredients(
                    generated_recipe["ingredients"],
                    pure_comma_separation(ingredients_list, return_list=True),

                    
                )
                directions = ext.directions(generated_recipe["directions"])
                image=load_image_from_local("asset\\images\\Capture.png")
                generated_recipe["by"] = chef

                r1, r2 = st.columns([6, 2])

                with r2:
                    recipe_post = generator.generate_frame(generated_recipe, chef.split()[-1])

                    st.image(
                        recipe_post,
                        caption="Save image and share on your social media",
                        use_column_width="auto",
                        output_format="PNG"
                    )

                with r1:
                    st.markdown(
                        " ".join([
                            "<div class='r-text-recipe'>",
                            "<div class='food-title'>",
                            f"<img src='{food_image}' />",
                            f"<h2 class='font-title text-bold'>{title}</h2>",
                            "</div>",
                            '<div class="divider"><div class="divider-mask"></div></div>',
                            "<h3 class='ingredients font-body text-bold'>Ingredients</h3>",
                            "<ul class='ingredients-list font-body'>",
                            " ".join([f'<li>{item}</li>' for item in ingredients]),
                            "</ul>",
                            "<h3 class='directions font-body text-bold'>Directions</h3>",
                            "<ol class='ingredients-list font-body'>",
                            " ".join([f'<li>{item}</li>' for item in directions]),
                            "</ol>",
                            '<div>'
                        ]),
                        unsafe_allow_html=True
                    )

if __name__ == '__main__':
    main()

