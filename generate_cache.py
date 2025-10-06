import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

#HandCrafted Prompts for editing concepts in CLIP Text Encoder

LabradorRetriever = [
    ("A Labrador Retriever standing calmly with a relaxed posture, its coat smooth and eyes gentle", "A domestic cat sitting quietly with a calm expression, its fur sleek and eyes alert"),
    ("A golden Labrador walking across an open field with a slow, steady pace", "A striped cat padding softly through an open field, tail held high"),
    ("A cheerful Labrador lying down on a patch of grass, tail twitching slightly", "A curious cat curled up on a patch of grass, eyes half-closed"),
    ("A Labrador Retriever looking up attentively, ears perked, ready for action", "A cat looking up with interest, ears turned forward, quietly focused"),
    ("A short-haired Labrador playfully nudging a toy across the ground", "A sleek cat pawing at a toy with quiet curiosity"),
    ("A light-coated Labrador resting beneath a tree on a warm afternoon", "A soft-furred cat napping under the shade of a tree"),
    ("A Labrador Retriever trotting along a dirt path with easy confidence", "A house cat strolling along a garden path with quiet grace"),
    ("A Labrador sitting at attention, gazing into the distance", "A cat perched on a ledge, observing its surroundings")
]
BarbetonDaisy = [
    ("A Barberton Daisy blooming brightly in the daylight", "A hibiscus blooming brightly in the daylight"),
    ("A single Barberton Daisy captured in soft natural light", "A single hibiscus captured in soft natural light"),
    ("A Barberton Daisy with dew drops on its petals", "A hibiscus with dew drops on its petals"),
    ("A Barberton Daisy swaying gently in the breeze", "A hibiscus swaying gently in the breeze"),
    ("A vibrant Barberton Daisy standing among green leaves", "A vibrant hibiscus standing among green leaves"),
    ("A freshly bloomed Barberton Daisy on a sunny day", "A freshly bloomed hibiscus on a sunny day"),
    ("A Barberton Daisy seen from above, showing its radiant symmetry", "A hibiscus seen from above, showing its radiant symmetry"),
    ("A Barberton Daisy growing beside a garden path", "A hibiscus growing beside a garden path")
]
BlueJay = [
    ("A blue bird perched on a tree branch", "A brown sparrow perched on a wooden fence"),
    ("A blue bird flying across a clear sky", "A brown sparrow fluttering near a garden hedge"),
    ("A blue bird resting on a windowsill", "A brown sparrow sitting quietly on a window ledge"),
    ("A blue bird singing in the morning light", "A brown sparrow chirping softly in a bush"),
    ("A blue bird near a small water fountain", "A brown sparrow hopping beside a puddle"),
    ("A blue bird sitting on a power line", "A brown sparrow balancing on a wire"),
    ("A blue bird feeding on sunflower seeds", "A brown sparrow pecking at bread crumbs"),
    ("A blue bird bathing in a birdbath", "A brown sparrow shaking off water beside the bath"),
    ("A blue bird watching from a tree canopy", "A brown sparrow peeking from under a leaf"),
    ("A blue bird flying over a grassy field", "A brown sparrow skimming low over dry grass"),
    ("A blue bird calling from a rooftop", "A brown sparrow perched on a chimney"),
    ("A blue bird surrounded by spring flowers", "A brown sparrow among fallen autumn leaves"),
    ("A blue bird in a snowy backyard", "A brown sparrow fluffed up in the cold"),
    ("A blue bird hopping on a garden path", "A brown sparrow darting across a patio"),
    ("A blue bird nesting in a birdhouse", "A brown sparrow gathering twigs for a nest"),
    ("A blue bird pausing on a park bench", "A brown sparrow resting on the armrest"),
    ("A blue bird silhouetted against a sunset", "A brown sparrow catching the last light of day"),
    ("A blue bird flapping its wings in excitement", "A brown sparrow stretching its wings after rest"),
    ("A blue bird exploring a backyard tree", "A brown sparrow searching for bugs in the bark"),
    ("A blue bird curiously watching from a branch", "A brown sparrow observing from a low shrub")
]
GolfBall = [
    ("A white golf ball resting on a flat surface","A yellow tennis ball resting on a flat surface"),
    ("A single golf ball with dimpled texture","A single tennis ball with fuzzy texture"),
    ("A golf ball placed under soft lighting","A tennis ball placed under soft lighting"),
    ("A close-up of a golf ball showing its dimples","A close-up of a tennis ball showing its seam"),
    ("A golf ball sitting next to a tee","A tennis ball sitting next to a racquet"),
    ("A clean golf ball on a neutral background","A clean tennis ball on a neutral background"),
    ("A slightly worn golf ball with minor scuffs","A slightly worn tennis ball with faint marks"),
    ("A golf ball balanced on grass","A tennis ball resting on a court")
]
AppleFruit = [
    ("A fresh red apple on a plain surface","A ripe orange resting on a flat surface"),
    ("A single apple with a smooth, shiny skin","A single orange with a textured peel"),
    ("An apple placed in soft natural light","An orange placed in soft natural light"),
    ("A close-up of an apple with a green leaf","A close-up of an orange with a leafy stem"),
    ("A whole apple with subtle color gradients","A whole orange with a vibrant peel"),
    ("An apple resting next to a small knife","An orange next to a peeled segment"),
    ("A red apple centered on a neutral background","An orange centered on a neutral background"),
    ("A slightly dewy apple sitting still","A slightly dewy orange sitting still")
]
VanGough = [
    ("an image of starry nights in Van Gogh style.", "A starlit sky above a quiet French village, glowing softly in the misty ambiance of Monet’s impressionist palette."),
    ("A vibrant vase of golden sunflowers bursting with texture and color in Van Gogh’s bold impasto technique.", "A soft, light-dappled bouquet of sunflowers in a sunlit room, captured in Monet’s gentle impressionist brushwork."),
    ("Café Terrace at Night", "An open-air café glowing under a soft starry sky, with gentle reflections and warm light in Monet’s impressionist style."),
    ("A windswept wheatfield under a stormy sky, dark crows in flight, painted with Van Gogh’s emotional urgency.", "A golden wheatfield under a cloudy sky with distant birds, captured in Monet’s soft, moody tones and light-filled textures."),
    ("An eerie café interior glowing with red and green hues, painted in Van Gogh’s intense emotional color palette.", "A quiet evening café bathed in diffused lamplight, with soft edges and gentle brushstrokes in Monet’s impressionist manner."),
    ("A vivid cluster of blooming irises, painted with energetic, swirling strokes and bold colors in Van Gogh’s signature style.", "A garden of soft violet irises swaying in the breeze, rendered in Monet’s shimmering light and delicate touch."),
    ("A cozy bedroom with simple furniture, depicted in Van Gogh’s flat perspective and bold color blocks.","A painting of cozy bedroom with simple furniture, rendered with soft, impressionist brushstrokes in the style of Claude Monet."),
    ("A jagged mountain glowing under a fiery dusk sky, with bold strokes and intense colors in Van Gogh’s impassioned style.","A distant mountain bathed in soft pink and lavender light at dusk, gently fading into the atmosphere in Monet’s impressionist style."),
    ("A bustling city street in the rain, with glistening reflections and thick swirling lines capturing movement and chaos in Van Gogh’s signature energy.","A rainy urban street softened by mist and glowing lamplight, painted with blurred reflections and delicate color washes in Monet’s touch."),
    ("A forest path bursting with fiery autumn leaves, painted in vivid oranges and rough, expressive brushstrokes in Van Gogh’s raw style.", "A peaceful forest path covered in fallen golden leaves, glowing under dappled sunlight in Monet’s airy impressionist rendering."),
    ("A rugged seaside cliff under a swirling sky, with crashing waves and thick, turbulent textures in Van Gogh’s emotionally intense style.", "A coastal cliff bathed in shimmering morning light, where the sea and sky melt together in soft hues à la Monet."),
    ("A dark lake beneath a glowing moon and swirling stars, with bold reflections and emotional contrasts in Van Gogh’s nocturnal palette.", "A quiet lake at night, where moonlight glimmers across the water’s surface in Monet’s subtle, impressionist light play."),
    ("A vast wheat field rippling in the wind, with a looming windmill and explosive sky in Van Gogh’s textured, symbolic brushwork.", "A golden wheat field under a gentle breeze, with a distant windmill dissolving into soft blues and yellows in Monet’s palette."),
    ("A scenery with expressive brushstrokes in Van Gogh’s raw style.","A scenery in Monet’s impressionist style."),
    ("A scenery with expressive brushstrokes in Van Gogh’s raw style.","A scenery in Monet’s impressionist style."),
]
Doodle = [
    ("A doodle of a flower with bold outlines and bright colors", "A blooming flower garden in soft hues in Monet’s style"),
    ("A simple doodle of a house with a chimney", "A countryside cottage bathed in golden light in Monet’s style"),
    ("A playful doodle of a cat sitting on a wall", "A cat resting near a garden wall in soft brushstrokes in Monet’s style"),
    ("A doodled tree with spiral leaves", "A tall tree with dappled sunlight in Monet’s impressionist brushwork"),
    ("A doodle of a sailboat on wavy water", "A sailboat gliding across shimmering water in Monet’s style"),
    ("A cartoon doodle of a mountain range", "Mountains in lavender and blue tones during dusk in Monet’s style"),
    ("A doodle sun with rays and a smiley face", "The sun rising over a misty landscape in Monet’s warm tones"),
    ("A child-like doodle of a garden with smiling flowers", "A flower-filled garden with soft color blends in Monet’s style"),
    ("A doodle of a park bench under a tree", "A peaceful park scene with a bench under dappled light in Monet’s style"),
    ("A doodled umbrella in the rain", "A figure with an umbrella walking in a rainy street in Monet’s impressionism"),
    ("A doodle of a fish swimming", "A koi fish pond with gentle ripples and light reflections in Monet’s palette"),
    ("A simple doodle of a bridge over a stream", "A curved bridge over lily-covered water in Monet’s garden at Giverny"),
    ("A doodle of a person holding balloons", "A child holding balloons in a sunlit field painted in Monet’s soft strokes"),
    ("A cartoonish doodle of ducks in a pond", "Ducks drifting on a calm pond surrounded by reeds in Monet’s style"),
    ("A whimsical doodle of a starry sky", "A night sky above a quiet village with soft starlight in Monet’s palette"),
    ("A line doodle of clouds and birds", "Birds flying across a hazy sky with pastel clouds in Monet’s brushwork"),
    ("A doodle of a windmill with spinning lines", "A windmill turning slowly in a golden wheat field in Monet’s colors"),
    ("A doodle of a street with tiny cars", "A village street under soft evening light in Monet’s impressionist view"),
    ("A scribbly doodle of a pond with frogs", "A reflective lily pond at dusk in Monet’s water garden style"),
    ("A doodled treehouse in a large tree", "A treehouse surrounded by spring foliage and filtered light in Monet’s style")
]
#Village
Neon = [
    ("A futuristic skyline glowing with neon lights and holograms", "A peaceful countryside with rolling green hills and stone cottages under blue skies"),
    ("A cyberpunk city skyline filled with towering skyscrapers and digital billboards", "A quiet village nestled among sunlit fields and clear skies"),
    ("A glowing skyline of a high-tech metropolis with electric mist", "A rural landscape with vineyards and a farmhouse under bright daylight"),
    ("A city skyline with flying cars weaving through neon-lit towers", "A scenic valley with grazing sheep and blue mountains in the distance"),
    ("A futuristic skyline shimmering in purple and blue neon at night", "A sunny hilltop with an old windmill overlooking flower fields"),
    ("A dense skyline with layered megastructures and animated advertisements", "A cozy European town with red-roofed houses and a clear sky"),
    ("A sprawling cyberpunk skyline beneath glowing drones and sky bridges", "A panoramic countryside view with golden wheat fields under sunlight"),
    ("A skyline of glowing vertical cities connected by light rails", "A sunlit road winding through lavender fields with blue skies above"),
    ("A nighttime city skyline with neon reflections on sleek glass towers", "A small lakeside village surrounded by green hills and white clouds"),
    ("A high-tech skyline dominated by spires and pulsing lights", "A bright vineyard-covered slope with a stone cottage and clear skies"),
    ("A massive futuristic skyline with floating structures and neon veins", "A quiet meadow with butterflies and blooming flowers beneath the sun"),
    ("A skyline under a digital aurora with colorful electric patterns", "A charming countryside lane with ivy-covered homes and blue skies"),
    ("A dark, glowing skyline with electric billboards stacked along buildings", "A scenic valley town surrounded by bright green fields and hills"),
    ("A skyline of glass towers lit in neon pinks and blues", "A quaint rural setting with a barn and sunny open pastures"),
    ("A cyberpunk skyline with giant robotic statues and light trails", "A peaceful hill with grazing cows and a small chapel under blue skies"),
    ("A neon skyline with tubes of light stretching across the sky", "A European farmstead with a vegetable garden and sun-drenched fields"),
    ("A futuristic skyline with airships hovering among neon towers", "A riverside village with a stone bridge and sunlight dancing on the water"),
    ("A vertical city skyline glowing with layered lights and motion", "A countryside railway curving through grassy plains and open skies"),
    ("A dense skyline with digital fog and neon energy pulses", "A green orchard with old trees and picnic tables under a blue sky"),
    ("A futuristic skyline surrounded by glowing rings and energy beams", "A sunlit path winding through a forest clearing in the countryside")
]
Monet = [
    ("A hazy sunrise over a misty harbor in Monet’s style", "A realistic sunrise over a foggy harbor with anchored boats"),
    ("A tranquil pond with floating lilies in Monet’s style", "A realistic pond with lily pads and clear reflections"),
    ("A graceful woman with a parasol in a flowery meadow in Monet’s style", "A realistic scene of a woman standing with a parasol in a blooming field"),
    ("Rouen Cathedral bathed in soft morning light in Monet’s style", "A realistic view of Rouen Cathedral in early sunlight"),
    ("A garden with a curved bridge and soft hues in Monet’s style", "A realistic garden with an arched bridge over still water"),
    ("A glowing haystack in warm sunset tones in Monet’s style", "A realistic haystack in a field at sunset"),
    ("The Houses of Parliament shrouded in purple mist in Monet’s style", "A realistic view of the Houses of Parliament on a foggy morning"),
    ("A mountain in soft pink and lavender at dusk in Monet’s style", "A realistic mountain scene lit by the colors of dusk"),
    ("A rainy urban street with soft lamplight in Monet’s style", "A realistic city street in the rain with glowing streetlights"),
    ("A forest path with golden leaves in dappled light in Monet’s style", "A realistic forest path covered with autumn leaves and filtered sunlight"),
    ("A coastal cliff in soft morning light in Monet’s style", "A realistic seaside cliff in early morning light with gentle waves below"),
    ("A quiet lake with moonlit glimmers in Monet’s style", "A realistic lake at night reflecting moonlight"),
    ("A golden wheat field with a distant windmill in Monet’s style", "A realistic wheat field swaying in the breeze with a windmill in the background"),
    ("A scenery in Monet’s style", "A realistic natural landscape"),
    ("A landscape in Monet’s style", "A realistic countryside view")
]
Sketch = [
    ("A fantasy castle drawn as an architectural concept sketch, with labeled sections and rough texture.", "A high-resolution photograph of fantasy castle with rough texture."),
    ("A pencil sketch", "A high-resolution photograph"),
    ("A sketch of a scenery", "A high-resolution photograph of a scenery"),
    ("A sketch of a landscape","A high-resolution photograph of a landscape"),
    ("A sketch of a potrait","A high-resolution photograph of a person"),
    ("A sketch of still life","A high-resolution photograph"),
    ("A pencil sketch", "A high-resolution photograph"),
]
Wedding = [
    ("A bride and groom exchanging vows in a flower-filled garden", "A family gathered in a park enjoying food and conversation"),
    ("A wedding ceremony under a decorated canopy with close relatives watching", "Relatives sitting under a tree, sharing stories and snacks on a sunny afternoon"),
    ("A groom tying a traditional wedding necklace on the bride", "A large family sitting together at a long dinner table, smiling and chatting"),
    ("A couple dressed in wedding attire standing on a stage surrounded by flowers", "An extended family enjoying a weekend lunch in a cozy home"),
    ("A joyful wedding baraat with music, dancing, and colorful attire", "Cousins dancing together at a family function in the backyard"),
    ("A traditional wedding ritual with the bride and groom seated near a sacred fire", "A family reunion with kids playing and adults relaxing on lawn chairs"),
    ("A wedding photo session with the couple posing in elegant clothes", "A family gathered for a group photo during a holiday event"),
    ("A newlywed couple being showered with rose petals", "A family enjoying a casual evening walk after dinner"),
    ("A bride adjusting her veil while bridesmaids stand nearby", "A group of siblings and cousins playing games indoors"),
    ("A grand wedding reception with lights, music, and celebration", "A family gathered in a decorated hall for a festive meal"),
    ("A couple exchanging rings in front of cheering guests", "A family celebrating a birthday with balloons, cake, and music"),
    ("An outdoor wedding with rows of seated guests and floral decorations", "A casual garden party where relatives mingle over drinks and snacks"),
    ("A bride walking down the aisle holding a bouquet", "Family members sitting together watching old photo albums"),
    ("A couple seated for a traditional wedding ritual with priests guiding them", "A cozy indoor dinner with grandparents, children, and relatives"),
    ("A close-up of a bride and groom smiling during their wedding portraits", "A family relaxing on a verandah with tea and laughter"),
    ("A destination wedding on a beach with the couple walking barefoot", "A family trip where members enjoy a picnic by the sea"),
    ("A couple dressed in ceremonial attire standing before a wedding altar", "A festive lunch gathering with relatives in traditional clothes"),
    ("Guests showering the couple with rice and blessings", "A family gathering around a bonfire telling stories and roasting snacks"),
    ("A wedding scene with rituals, garlands, and joyful music", "Relatives cooking together in a bustling kitchen before a celebration"),
    ("A couple seated on a decorated swing during a wedding celebration", "A lively indoor celebration where kids are running around and adults chat")
]
Sunset = [
    ("Sunset casting fiery hues over a winding mountain road", "Mountain road bathed in bright afternoon blue sky"),
    ("Dusky glow over a tranquil lakeside dock at sunset", "Tranquil lakeside dock under a clear afternoon blue sky"),
    ("Crimson sunset illuminating jagged desert canyons", "Jagged desert canyons under a cloudless afternoon blue sky"),
    ("Golden hour light spilling through a dense pine forest", "Dense pine forest beneath a bright afternoon blue sky"),
    ("Sunset reflection shimmering on a calm river bend", "Calm river bend under a crisp afternoon blue sky"),
    ("Silhouetted city skyline against a vibrant sunset sky", "City skyline beneath a clear afternoon blue sky"),
    ("Lavender and orange sunset over rolling vineyard hills", "Rolling vineyard hills under a sunny afternoon blue sky"),
    ("Sunset glow bathing a seaside lighthouse in warm light", "Seaside lighthouse under a bright afternoon blue sky"),
    ("Warm sunset tones highlighting rocky coastal cliffs", "Rocky coastal cliffs under a cloudless afternoon blue sky"),
    ("Sunset light filtering through autumn forest foliage", "Autumn forest foliage under a clear afternoon blue sky"),
    ("Fiery sunset over a sprawling urban park", "Sprawling urban park beneath a crisp afternoon blue sky"),
    ("Sunset haze enveloping a winding marshland boardwalk", "Winding marshland boardwalk under a bright afternoon blue sky"),
    ("Soft sunset pastels above a serene waterfall plunge pool", "Serene waterfall plunge pool under a clear afternoon blue sky"),
    ("Sunset silhouette of a lone cactus on rocky desert terrain", "Lone cactus on rocky desert terrain under a cloudless afternoon blue sky"),
    ("Orange sunset backlighting a rustic rural barn", "Rustic rural barn under a vivid afternoon blue sky"),
    ("Sunset radiance over snow-capped alpine peaks", "Snow-capped alpine peaks beneath a clear afternoon blue sky"),
    ("Sunset glow washing over a meandering mountain stream", "Meandering mountain stream under a bright afternoon blue sky"),
    ("Sunset colors streaking across a verdant meadow", "Verdant meadow beneath a clear afternoon blue sky"),
    ("Molten sunset light cascading down a rocky waterfall", "Rocky waterfall under a crisp afternoon blue sky"),
    ("Sunset’s last rays illuminating an ancient stone bridge", "Ancient stone bridge under a serene afternoon blue sky")
]
Rainfall = [
    ("A countryside road glistening under a gentle rain", "A dusty rural road baking under a scorching sun"),
    ("A green meadow soaked in steady rainfall", "A cracked, sun-bleached meadow with dry grass"),
    ("Rain falling over a bustling market with umbrellas open", "An overheated market square with people wiping sweat in the heat"),
    ("A forest shimmering with rain-drenched leaves", "A forest with withered trees and dry, brittle foliage under intense heat"),
    ("Children dancing in puddles on a rainy afternoon", "Children walking slowly across a dry, barren field under harsh sun"),
    ("A muddy path through the woods after rain", "A parched forest trail covered in dust and fallen dry leaves"),
    ("Raindrops splashing in a quiet village pond", "A cracked, dried-up pond surrounded by wilting vegetation"),
    ("A cityscape blurred by falling rain and headlights", "A sunbaked city street radiating heat under a cloudless sky"),
    ("A farmer's field rich with rain-soaked soil", "A sun-scorched field with dry, hardened earth and wilting crops"),
    ("A lush hilltop with rain clouds overhead", "A dry hilltop under blazing sun with sparse, yellowing grass"),
    ("A calm lake rippling under gentle rainfall", "A dry lakebed cracked and exposed to relentless sun"),
    ("A mountain valley misted with fresh rain", "A barren mountain slope under intense heat and dust storms"),
    ("A neighborhood garden thriving in a light drizzle", "A sun-dried garden with curling leaves and dusty air"),
    ("Heavy rain falling on a tin-roofed home", "The tin roof gleaming hot under midday sun in a drought-struck area"),
    ("A rain-soaked trail through thick greenery", "A dry hiking trail surrounded by scorched vegetation"),
    ("A village wrapped in soft rain and cool breeze", "A rural village baking in dry heat with no sign of clouds"),
    ("A street full of puddles reflecting gray skies", "A cracked street under a fierce sun and rising heat haze"),
    ("Rainwater dripping from leafy branches in a forest", "Shriveled branches under intense sunlight in a dry forest"),
    ("A quiet rainy park with wet benches and fresh air", "A deserted park with yellowed grass and dry dust blowing in the wind"),
    ("Monsoon clouds pouring rain over a coastal town", "A coastal town sweltering in extreme heat and dry sea winds")
]
AuroraBorialis = [
    ("Photo-realistic aurora borealis over a frozen lake", "Photo-realistic frozen lake under a soft sunrise sky with warm orange and pink hues"),
    ("A cabin under the glowing northern lights", "A cabin bathed in the gentle light of early morning as the sun rises"),
    ("Mountains silhouetted beneath the aurora", "Mountains glowing under the first light of sunrise, casting long shadows"),
    ("A forest glowing beneath the northern lights", "A forest touched by golden sunlight breaking through the morning mist"),
    ("Photo-realistic sky filled with green aurora waves", "Photo-realistic sky painted with pastel tones of a calm sunrise"),
    ("Northern lights dancing above a snowy field", "Snowy field bathed in soft golden and pink sunrise light"),
    ("A lone traveler watching the aurora borealis", "A lone traveler walking beneath a colorful sunrise sky"),
    ("Vibrant auroras over a quiet arctic village", "A quiet arctic village under the warm, rising sun with glowing rooftops"),
]
#
Scenery = [
    ("A green forest with sunlight filtering through the canopy", "A dry, barren landscape with cracked soil and no trees"),
    ("A peaceful valley filled with wildflowers and tall grass", "A wide, desolate plain with no vegetation in sight"),
    ("A scenic hillside covered in lush greenery", "A dusty hillside with bare rocks and no plant life"),
    ("A tranquil meadow with colorful flowers and buzzing bees", "A flat, lifeless field with dry dirt and no trees"),
    ("A forest path surrounded by thick trees and birdsong", "A dirt path cutting through a treeless wasteland"),
    ("Rolling green hills under a bright blue sky", "Rolling dusty hills with dry earth and no greenery"),
    ("A river winding through a dense forest", "A dried-up riverbed cutting through a barren landscape"),
    ("A garden filled with blooming plants and shady trees", "An empty yard with cracked ground and no vegetation"),
    ("A lush countryside dotted with trees and shrubs", "An open stretch of land stripped bare of trees and plants"),
    ("A vibrant mountain valley with tall pines", "A rocky valley where not a single tree stands"),
    ("A peaceful park with trees casting long shadows", "An open field scorched by the sun, with nothing but dirt"),
    ("A lakeside surrounded by forest", "A dry lakebed surrounded by barren, treeless land"),
    ("A sunlit glade in the woods", "A sun-bleached clearing with nothing but sand and rock"),
    ("A shady grove filled with birds and undergrowth", "A wide expanse of dry land devoid of any trees"),
    ("A mountain trail lined with evergreens", "A mountain slope of bare stone and sparse scrub"),
    ("A riverside walk with trees arching overhead", "A deserted riverbank with dry soil and no tree cover"),
    ("A serene forest clearing", "An empty plain of hard earth and scattered stones"),
    ("A grassy hilltop with views of distant forests", "A bald hilltop stripped of vegetation under harsh sunlight"),
    ("A coastal path with sea breeze and pine trees", "A barren coastal stretch with exposed soil and no shelter"),
    ("A springtime orchard in bloom", "A dry, cracked field where no trees grow")
]
Sleeping = [
    ("A boy sleeping peacefully","A boy sitting with arms folded, smiling proudly"),
    ("A girl sleeping quietly","A girl sitting with arms folded, smiling playfully"),
    ("A person sleeping calmly","A person sitting with arms folded, smiling proudly"),
    ("A man sleeping soundly","A man sitting with arms folded, smiling proudly"),
    ("A woman sleeping peacefully","A woman sitting with folded arms and a warm smile"),
    ("A boy sleeping deeply","A boy sitting with arms folded, smiling confidently"),
    ("A girl sleeping with a gentle expression","A girl sitting with folded arms and a playful smile"),
    ("A woman sleeping silently","A woman sitting with folded arms, smiling peacefully")
]
Walking = [
    ("A boy walking on a sunny afternoon","A boy sitting with arms folded, smiling proudly"),
    ("A girl walking through a park","A girl sitting with arms folded, smiling playfully"),
    ("A person walking down a quiet street","A person sitting with arms folded, smiling proudly"),
    ("A man walking in the morning light","A man sitting with arms folded, smiling proudly"),
    ("A woman walking along a garden path","A woman sitting with folded arms and a warm smile"),
    ("A boy walking beside a calm river","A boy sitting with arms folded, smiling confidently"),
    ("A girl walking across an open field","A girl sitting with folded arms and a playful smile"),
    ("A woman walking past a large window",  "A woman sitting with folded arms, smiling peacefully")
]
Eating = [
    ("A boy eating a delicious meal", "A boy walking down a quiet road"),
    ("A girl eating a tasty snack", "A girl walking along a tree-lined road"),
    ("A person eating a hearty meal", "A person walking on an empty road"),
    ("A man eating a satisfying breakfast", "A man walking down a paved road"),
    ("A woman eating a fresh lunch", "A woman walking along a peaceful road"),
    ("A boy eating a warm dinner", "A boy walking slowly on a narrow road"),
    ("A girl eating a sweet treat", "A girl walking on a sunny road"),
    ("A woman eating a delightful meal", "A woman walking down a quiet country road")
]
#Meditating 
Dancing = [
    ("A boy dancing joyfully","A boy sitting with arms folded, smiling proudly"),
    ("A girl dancing playfully","A girl sitting with arms folded, smiling playfully"),
    ("A person dancing freely","A person sitting with arms folded, smiling proudly"),
    ("A man dancing confidently","A man sitting with arms folded, smiling proudly"),
    ("A woman dancing elegantly","A woman sitting with folded arms and a warm smile"),
    ("A boy dancing energetically","A boy sitting with arms folded, smiling confidently"),
    ("A girl dancing with excitement","A girl sitting with folded arms and a playful smile"),
    ("A woman dancing gracefully","A woman sitting with folded arms, smiling peacefully")
]
Jumping = [
    ("A boy jumping energetically", "A boy sitting with arms folded, smiling proudly"),
    ("A girl jumping playfully", "A girl sitting with arms folded, smiling playfully"),
    ("A person jumping freely", "A person sitting with arms folded, smiling calmly"),
    ("A man jumping confidently", "A man sitting with arms folded, smiling confidently"),
    ("A woman jumping gracefully", "A woman sitting with arms folded, smiling warmly"),
    ("A boy jumping with excitement", "A boy sitting with arms folded, grinning with excitement"),
    ("A girl jumping with joy", "A girl sitting with arms folded, smiling joyfully"),
    ("A woman jumping lightly", "A woman sitting with arms folded, smiling peacefully")
]

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
tokenizer = pipe.tokenizer

concept_name = ["LabradorRetriever", "BarbetonDaisy", "BlueJay", "GolfBall", "AppleFruit", "VanGough", "Doodle", "Neon", "Monet", "Sketch", "Wedding", "Sunset", "Rainfall", "AuroraBorialis", "Scenery", "Sleeping", "Walking", "Eating", "Dancing", "Jumping"]
concept_list = [LabradorRetriever, BarbetonDaisy, BlueJay, GolfBall, AppleFruit, VanGough, Doodle, Neon, Monet, Sketch, Wedding, Sunset, Rainfall, AuroraBorialis, Scenery, Sleeping, Walking, Eating, Dancing, Jumping]

token_dict = {}
for c_name in tqdm(concept_name):
    token_dict[c_name] = {
        "concept": [],
        "edit": []
    }

for i,c_list in enumerate(concept_list):
    concept_prompt = [c[0] for c in c_list]
    edit_prompt = [c[1] for c in c_list]

    concept_tokenized = tokenizer(
        concept_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    edit_tokenized = tokenizer(
        edit_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    token_dict[concept_name[i]]["concept"] = concept_tokenized
    token_dict[concept_name[i]]["edit"] = edit_tokenized

torch.save(token_dict, "token_dict")