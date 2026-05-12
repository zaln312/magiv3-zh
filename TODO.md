我想做一个magi studio系统，包含简单前后端。

主要流程（部分代码见main.py）：

1、上传图片。得到img_paths[img_idx]=list[str]。

2、ocr识别，调用paddleocr模型。得到unordered_ocr_res[img_idx]=dict{"boxes"文本框，"texts"文本框对应的文字}。

3、predict，调用magiv3模型。得到results[img_idx]=dict{'panels', 'texts', 'characters', 'tails', 'ocr_texts', 'text_panel_associations', 'text_character_associations', 'text_tail_associations'，
'global_character_ids'}

4、caption，调用llamacpp模型。对每个图片内的每个panel生成一个描述。

5、character grounding，调用magiv3模型；caption 插入 [id]。这两个子步骤综合得到grounded_captions，每一个panel对应一个描述，在人称代词处插入[其对应id]。

6、panel_scripts。结合results，构建panel级别的对白脚本。

7、prose_prompt。结合grounded_captions + panel_scripts，构建panel级别的画面描述+对白脚本。

8、prose，调用llamacpp模型。根据prose_prompt，生成全部panel范围的（即包含全部图片的），流畅完整叙述性文本。

这是一个围绕magiv3的流程，除了magiv3的florence2模型以外，其他模型不应该在这个系统的设计范畴里，应该依赖用户实现api的调用，（但本系统提供默认的api调用）。
系统本身不提供的，其他模型的服务，具体来说：
get_captions、get_prose_prompt、get_prose需要大语言模型。get_ocr_results需要文本检测+识别模型。以上四个方法会内部调用“支持用户自行实现api调用”的方法，例如user_get_ocr_results（ocr api 调用）；且get_ocr_results内部有“支持用户自行实现格式转换”的方法 user_format_ocr_results（ocr results 格式化）。
此外系统内部有严格check_format用于校验 ocr results 格式。

由于所有模型都是串行调用的，对于调用本地模型的情况（例如本系统提供的默认api调用 paddleocr和llamacpp），支持串行实时加载、卸载 当前需要用到的模型，以降低显存负担；也支持先全部加载完毕，不做动态加载、卸载。

用户的交互功能，每个功能独立页面，除了下面的功能不额外增加：

1、上传图片（支持多张上传）后，展示。支持选择删除，拖动排序。

2、ocr识别后，对于每张图片，展示识别到的文本框box，框拉出虚线到画面外，与展示对应的文本内容的框text相连。
box支持删除（同步删除对应text），拖动左上坐标、右下坐标 调整box。
text支持直接在框内编辑文本内容。

3、predict后，对于每张图片，展示人物框character；通过text_character_associations关联的文本框box，通过虚线与文本框box相连。
character支持删除（如有，同步删除其与文本框box的全部关联），支持拖动左上坐标、右下坐标 调整character框，支持添加人物框（新增人物框无需重新跑 association）。
点击人物框，通过勾选或取消勾选box，来更改其关联的文本框。

全局增加一个维护 global_character_ids -> 角色名 的映射表变量，角色名初始均为空。（最终会写入panel_scripts中）

图片外固定位置展示一个角色列表，每行包含：一个角色的character图片（取上传的全部图片范围内，最大尺寸的人物图片）、global_character_ids、该id对应的角色名。

支持人物聚类微调功能。点击人物框，角色列表滚动定位到该人物的行，支持修改global_character_ids，修改角色名（即修改映射表变量）。

5、character grounding后，对于一张图片，上面的每个panel框通过虚线，与图片外的grounded_caption框相连；图片中有为id分配不同颜色的角色框。grounded_caption框展示该panel的文本描述，人称代词处的[其对应id]使用颜色，与角色框颜色对应。

7、prose_prompt构建后，文本展示。

8、prose生成后，文本展示。

先综合以上需求，编写任务书。
任务可以从粗到细、从大到小设计，每一步实现后，待我运行确认无误，再开始下一步。不要一步到位，避免大量问题同时出现，难以调试。