我想做一个magi studio系统，包含简单前后端。

前端Vue，搭配现成的 UI 组件库，需要较强的画布交互能力。
后端使用 Python 的轻量 Web 框架

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

由于所有模型都是串行调用的，对于调用本地模型的情况（例如本系统提供的默认api调用 paddleocr （ocr server已外部实现）和llamacpp），支持串行实时加载、卸载 当前需要用到的模型，以降低显存负担；也支持先全部加载完毕，不做动态加载、卸载。

用户的交互功能，每个功能独立页面，除了下面的功能不额外增加：

1、上传图片（支持多张上传）后，展示。支持选择删除，拖动排序。

2、ocr识别后，对于每张图片，展示识别到的文本框box，框外有id标记，与文本内容列表的条目text对应。
box支持删除（同步删除对应text），支持拖动左上坐标、右下坐标 调整box。
text支持直接在框内编辑文本内容。

3、predict后，对于每张图片，展示人物框character；通过text_character_associations关联的文本框box，通过虚线与文本框box相连。
character支持删除（如有，同步删除其与文本框box的全部关联），支持拖动左上坐标、右下坐标 调整character框，支持添加人物框（新增人物框无需重新跑 association）。
点击人物框，通过勾选或取消勾选box，来更改其关联的文本框。

全局增加一个维护 global_character_ids -> 角色名 的映射表变量，角色名初始均为空。（最终会写入panel_scripts中）

图片外固定位置展示一个角色列表，每行包含：一个角色的character图片（取上传的全部图片范围内，最大尺寸的人物图片）、global_character_ids、该id对应的角色名。

支持人物聚类微调功能。点击人物框，角色列表滚动定位到该人物的行，支持修改global_character_ids，修改角色名（即修改映射表变量）。

5、character grounding后，对于每张图片，上面的一个panel框，对应右侧caption列表的一个grounded_caption文本框；图片中有为id分配不同颜色的角色框。grounded_caption框展示该panel的文本描述，人称代词处的[其对应id]使用颜色，与角色框颜色对应。

7、prose_prompt构建后，文本展示。

8、prose生成后，文本展示。

先综合以上需求，制定任务流程。
任务可以从粗到细、从大到小设计，每一步实现后，待我运行确认无误，再开始下一步。不要一步到位，避免大量问题同时出现，难以调试。


------------------------------------------------------


把人物参考图页面 写成和 Predict 预测、Grounding 指代、Prose 叙述 等等 同级别的主页面。
在 Prose 叙述 页面点击下一步进入 人物参考图页面。
下拉菜单支持选择不同的参考图来源，切换到对应页面：
1、角色设定集，用户自行上传图片
2、用户自行挑选（待选队列按尺寸排序，现有逻辑）人物裁剪图片，调用参考图生成模型，生成参考图。支持重新生成，不断扩充待选的参考图队列（同上的队列逻辑，但这个队列支持一键清空）。
以上两个选项对应页面的结构，都参照现有的人物参考图界面，每行一个角色。
为参考图生成模型添加设置页面，同样通过json配置


prose叙述 添加一个新子页面“故事背景描述”在两个子页面之前，输入故事的背景描述。添加到prose_prompt的最前面。
构造出结构：
"Story background:",
"{用户输入的故事背景描述}",
"",
"Next is a series of manga panel descriptions and dialogues."
(后面同原结构)


添加最后一个主页面 视频生成。
在 人物参考图 页面点击下一步进入 视频生成页面。
第一个子页面，左半展示prose（支持一键复制）,右半按行展示每个角色的参考图（支持一键打包导出）。点击下一步进入第二个子页面。
第二个子页面调用视频生成模型，展示生成的视频。


json文件的必填内容都应该在保存时做校验。


为项目中的caption生成 使用的文字图片生成文字模型、prose生成 使用的文字生文字模型、人物参考图生成 使用的文字图片生成图片模型、视频生成 使用的 图片文字生成视频 模型，分别编写用于测试系统的fake server 独立脚本，每个server把请求和响应都格式化打印，用于debug。
并告诉我对应设置页面的json填写内容。    


-------------------------------------------------


task_id
string 
必需
Vidu 生成的任务ID
state
string 
必需
处理状态
可选值：
created 创建成功
queueing 任务排队中
processing 任务处理中
success 任务成功
failed 任务失败

vidu q3的基础工作流程（其他视频生成服务应该也类似）
1.认证（Authentication）
所有请求均需在请求头(Header)中携带：
Token 用于身份验证的 API 密钥
其他生成任务所需的参数
2.提交请求
将生成任务提交到对应的接口地址： `https://api.vidu.cn` 
3.响应处理
提交成功后，响应体中会返回一个task_id，用于后续状态查询。
4.状态查询
使用返回的 task_id 定期轮询视频生成状态：
GET `https://api.vidu.cn/ent/v2/tasks/{id}/creations` 
5.获取结果
当处理完成后，状态接口将返回 status = success，此时可在响应中获取完整视频 URL 及其他元数据。
  "id":"your_task_id",
  "state": "success",
  "err_code": "",
  "credits": 4,
  "payload": "",
  "creations": [
    {
      "id": "your_creations_id",
      "url": "your_generated_results_url",
      "cover_url": "your_generated_results_cover_url"
    }
  ]
}
为了让视频生成的调用相关逻辑独立（减少系统集成，便于更换其他视频模型），必要的信息（如最基本的轮询获取状态方法、解析响应、获取视频、状态类型等）让用户在系统设置 - 视频生成页面自行填写json或python。
支持用户点击生成视频后返回首页，项目状态上同步显示当前状态（已有逻辑，可能需完善）
