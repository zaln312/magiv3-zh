<template>
  <div class="settings-page">
    <div class="page-header">
      <h2>系统设置</h2>
    </div>

    <el-card v-loading="loading" class="settings-card">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="Magi V3 模型" name="magi">
          <div class="tab-content">
            <el-form label-position="top">
              <el-form-item label="加载模式">
                <div class="form-item-block">
                  <el-radio-group v-model="config.magi_v3_mode" @change="onMagiModeChange">
                    <el-radio label="dynamic">动态加载卸载</el-radio>
                    <el-radio label="persistent_project">进入项目后保持加载（退出首页后卸载）</el-radio>
                  </el-radio-group>
                  <div class="help-text">
                    <p><strong>动态加载卸载</strong>：每次调用时加载，完成后立即卸载。节省显存，但每次操作有加载延迟。</p>
                    <p><strong>保持加载</strong>：进入项目时加载，直到返回首页才卸载。显存占用高，但所有操作更快。</p>
                  </div>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="OCR 设置" name="ocr">
          <div class="tab-content">
            <el-form label-position="top">
              <el-form-item label="API 地址">
                <el-input v-model="config.ocr.api_url" placeholder="http://127.0.0.1:8000/ocr" clearable />
              </el-form-item>
              <el-form-item label="user_format_ocr_results — 返回值格式化代码">
                <div class="form-item-block">
                  <div class="code-editor-wrapper">
                    <textarea
                      v-model="ocrFormatCode"
                      class="code-editor"
                      spellcheck="false"
                      rows="18"
                      placeholder="必须定义 user_format_ocr_results(results) 函数 ..."
                    ></textarea>
                    <el-button
                      size="small"
                      text
                      class="reset-btn"
                      @click="resetOcrFormatCode"
                    >
                      恢复默认
                    </el-button>
                  </div>
                  <div class="help-text">
                    <p>必须定义函数 <code>user_format_ocr_results(results)</code>，接收原始 OCR 结果列表，返回格式化的列表。</p>
                    <p>返回值需过系统 <code>check_format</code> 校验（每个元素需含 <code>polys</code>, <code>texts</code> 字段）。</p>
                  </div>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="Caption 生成" name="caption">
          <div class="tab-content">
            <el-form label-position="top">
              <el-form-item label="完整配置 (JSON)">
                <div class="form-item-block">
                  <div class="code-editor-wrapper">
                    <textarea
                      v-model="captionJson"
                      class="code-editor"
                      spellcheck="false"
                      rows="24"
                      placeholder="{ &quot;base_url&quot;: &quot;...&quot;, ... }"
                    ></textarea>
                    <el-button
                      size="small"
                      text
                      class="reset-btn"
                      @click="resetCaptionJson"
                    >
                      恢复默认
                    </el-button>
                  </div>
                  <el-alert
                    v-if="captionJsonError"
                    type="error"
                    :title="captionJsonError"
                    :closable="false"
                    style="margin-top: 8px"
                  />
                  <div class="help-text">
                    <p>OpenAI-compatible API。必填字段：<code>base_url</code>, <code>api_key</code>, <code>model</code>, <code>prompt_template</code>。</p>
                    <p>可选参数：<code>temperature</code>, <code>top_p</code>, <code>max_tokens</code>, <code>presence_penalty</code>, <code>extra_body</code>。</p>
                  </div>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="Prose 生成" name="prose">
          <div class="tab-content">
            <el-form label-position="top">
              <el-form-item label="完整配置 (JSON)">
                <div class="form-item-block">
                  <div class="code-editor-wrapper">
                    <textarea
                      v-model="proseJson"
                      class="code-editor"
                      spellcheck="false"
                      rows="24"
                      placeholder="{ &quot;base_url&quot;: &quot;...&quot;, ... }"
                    ></textarea>
                    <el-button
                      size="small"
                      text
                      class="reset-btn"
                      @click="resetProseJson"
                    >
                      恢复默认
                    </el-button>
                  </div>
                  <el-alert
                    v-if="proseJsonError"
                    type="error"
                    :title="proseJsonError"
                    :closable="false"
                    style="margin-top: 8px"
                  />
                  <div class="help-text">
                    <p>OpenAI-compatible API。必填字段：<code>base_url</code>, <code>api_key</code>, <code>model</code>, <code>prompt_template</code>。</p>
                    <p><code>prompt_template</code> 中 <code>{prompt}</code> 不可删除，否则无法注入面板描述与对话内容。</p>
                  </div>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="参考图生成" name="reference">
          <div class="tab-content">
            <el-form label-position="top">
              <el-form-item label="完整配置 (JSON)">
                <div class="form-item-block">
                  <div class="code-editor-wrapper">
                    <textarea
                      v-model="referenceJson"
                      class="code-editor"
                      spellcheck="false"
                      rows="24"
                      placeholder="{ &quot;enabled&quot;: true, ... }"
                    ></textarea>
                    <el-button
                      size="small"
                      text
                      class="reset-btn"
                      @click="resetReferenceJson"
                    >
                      恢复默认
                    </el-button>
                  </div>
                  <el-alert
                    v-if="referenceJsonError"
                    type="error"
                    :title="referenceJsonError"
                    :closable="false"
                    style="margin-top: 8px"
                  />
                  <div class="help-text">
                    <p>参考图生成模型配置 (IP-Adapter + SD1.5)。</p>
                    <p>必填字段：<code>enabled</code>。</p>
                    <p><code>prompt_template</code> 中 <code>{view}</code> 和 <code>{character_name}</code> 为视图和角色名占位符。</p>
                    <p>可选参数：<code>base_url</code>, <code>api_key</code>, <code>model</code>, <code>negative_prompt</code>, <code>num_images_per_view</code>, <code>width</code>, <code>height</code>。</p>
                  </div>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <el-tab-pane label="视频生成" name="video">
          <div class="tab-content">
            <el-form label-position="top">
              <el-form-item label="完整配置 (JSON)">
                <div class="form-item-block">
                  <div class="code-editor-wrapper">
                    <textarea
                      v-model="videoJson"
                      class="code-editor"
                      spellcheck="false"
                      rows="24"
                      placeholder="{ &quot;enabled&quot;: true, ... }"
                    ></textarea>
                    <el-button
                      size="small"
                      text
                      class="reset-btn"
                      @click="resetVideoJson"
                    >
                      恢复默认
                    </el-button>
                  </div>
                  <el-alert
                    v-if="videoJsonError"
                    type="error"
                    :title="videoJsonError"
                    :closable="false"
                    style="margin-top: 8px"
                  />
                  <div class="help-text">
                    <p>视频生成模型配置（通用任务提交+轮询工作流）。适用于 Vidu Q3 及类似异步视频生成 API。</p>
                    <p>必填字段：<code>enabled</code>, <code>submit.url</code>, <code>poll.url_template</code>。</p>
                    <p><strong>submit</strong>：提交任务配置。<code>body_template</code> 中支持占位符 <code>{prose}</code>（Prose叙述文本）和 <code>{images_base64}</code>（角色参考图 Base64 数组）。</p>
                    <p><strong>poll</strong>：轮询状态配置。<code>url_template</code> 中 <code>{task_id}</code> 会自动替换。<code>state_values</code> 用于映射 API 返回的状态字符串。</p>
                    <p><strong>submit_code / poll_code / parse_code</strong>：可选，Python 代码覆盖（函数名分别为 <code>user_submit_video</code>, <code>user_poll_video</code>, <code>user_parse_video</code>）。</p>
                  </div>
                </div>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>
      </el-tabs>

      <div class="actions">
        <el-button @click="handleReset" :loading="saving">恢复全部默认设置</el-button>
        <el-button type="primary" @click="handleSave" :loading="saving">保存设置</el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { configApi, projectApi } from '../api/endpoints'

const router = useRouter()

interface AppConfig {
  magi_v3_mode: 'dynamic' | 'persistent_project'
  ocr: {
    api_url: string
    format_code: string
  }
  caption: Record<string, any>
  prose: Record<string, any>
  reference: Record<string, any>
  video: Record<string, any>
}

const loading = ref(true)
const saving = ref(false)
const activeTab = ref('magi')

const config = ref<AppConfig>({
  magi_v3_mode: 'dynamic',
  ocr: {
    api_url: 'http://127.0.0.1:8000/ocr',
    format_code:
      'def user_format_ocr_results(results):\n' +
      '    """\n' +
      '    用户提供的 ocr results 格式化\n' +
      '    """\n' +
      '    return [\n' +
      '        {\n' +
      '            "polys": res["rec_polys"],\n' +
      '            "texts": res["rec_texts"],\n' +
      '        }\n' +
      '        for res in results\n' +
      '    ]\n',
  },
  caption: {},
  prose: {},
  reference: {},
  video: {},
})

const ocrFormatCode = ref('')
const captionJson = ref('')
const proseJson = ref('')
const referenceJson = ref('')
const videoJson = ref('')
const captionJsonError = ref('')
const proseJsonError = ref('')
const referenceJsonError = ref('')
const videoJsonError = ref('')

const ocrDefaultFormatCode =
  'def user_format_ocr_results(results):\n' +
  '    """\n' +
  '    用户提供的 ocr results 格式化\n' +
  '    """\n' +
  '    return [\n' +
  '        {\n' +
  '            "polys": res["rec_polys"],\n' +
  '            "texts": res["rec_texts"],\n' +
  '        }\n' +
  '        for res in results\n' +
  '    ]\n'

const captionDefault: Record<string, any> = {
  base_url: 'http://localhost:8001/v1',
  api_key: 'EMPTY',
  model: 'Qwen3.5-4B',
  prompt_template:
    'Describe this image in a single prose paragraph. ' +
    'For each character, start by clearly stating their relative position ' +
    "(e.g., 'the character on the left', 'in the foreground', 'the girl on the right'), " +
    'then describe their appearance (hair, clothing), and finally their actions or emotions. ' +
    'Do not use specific names. Ignore all embedded text, speech bubbles, and dialogue. ' +
    'Focus purely on visual elements.',
  temperature: 0.7,
  top_p: 0.8,
  max_tokens: 1024,
  presence_penalty: 1.5,
  extra_body: {
    top_k: 20,
    chat_template_kwargs: { enable_thinking: false },
  },
}

const proseDefault: Record<string, any> = {
  base_url: 'http://localhost:8001/v1',
  api_key: 'EMPTY',
  model: 'Qwen3.5-4B',
  prompt_template:
    '{prompt}\n\n' +
    'I want you to write a summary in Chinese so that a blind or visually impaired person can understand the story. ' +
    'Make sure to stick to the provided details. All these panels belong to the same page so make sure your narrative is coherent. ' +
    'The format of the narrative should be a prose.',
  temperature: 0.7,
  top_p: 0.8,
  max_tokens: 4096,
  presence_penalty: 1.5,
  extra_body: {
    top_k: 20,
    chat_template_kwargs: { enable_thinking: false },
  },
}

const referenceDefault: Record<string, any> = {
  enabled: true,
  base_url: 'http://localhost:8001/v1',
  api_key: 'EMPTY',
  model: '',
  prompt_template:
    'character reference sheet, {view} view, full body standing pose, ' +
    '{character_name}, clean white background, anime manga style, ' +
    'detailed character design, high quality, professional illustration',
  negative_prompt:
    'blurry, low quality, distorted face, bad anatomy, extra limbs, ' +
    'missing limbs, deformed hands, watermark, text, signature',
  num_images_per_view: 1,
  width: 512,
  height: 768,
}

const videoDefault: Record<string, any> = {
  enabled: true,
  submit: {
    url: '',
    method: 'POST',
    headers: {},
    body_template: {},
    task_id_path: 'id',
  },
  poll: {
    url_template: '',
    method: 'GET',
    headers: {},
    interval_seconds: 5,
    max_attempts: 120,
    state_path: 'state',
    state_values: {
      created: 'created',
      queueing: 'queueing',
      processing: 'processing',
      success: 'success',
      failed: 'failed',
    },
    creations_path: 'creations',
    url_path: 'url',
    cover_url_path: 'cover_url',
  },
  submit_code: '',
  poll_code: '',
  parse_code: '',
}

function prettyJson(obj: Record<string, any>): string {
  return JSON.stringify(obj, null, 2)
}

function syncEditors() {
  ocrFormatCode.value = config.value.ocr.format_code || ocrDefaultFormatCode
  captionJson.value = prettyJson(config.value.caption)
  proseJson.value = prettyJson(config.value.prose)
  referenceJson.value = prettyJson(config.value.reference || referenceDefault)
  videoJson.value = prettyJson(config.value.video || videoDefault)
}

async function loadConfig() {
  loading.value = true
  try {
    const res = await configApi.get()
    if (res.data.success && res.data.config) {
      Object.assign(config.value, res.data.config)
    }
    syncEditors()
  } catch (e: any) {
    ElMessage.error('加载配置失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    loading.value = false
  }
}

function resetOcrFormatCode() {
  ocrFormatCode.value = ocrDefaultFormatCode
}

function resetCaptionJson() {
  captionJson.value = prettyJson(captionDefault)
  captionJsonError.value = ''
}

function resetProseJson() {
  proseJson.value = prettyJson(proseDefault)
  proseJsonError.value = ''
}

function resetReferenceJson() {
  referenceJson.value = prettyJson(referenceDefault)
  referenceJsonError.value = ''
}

function resetVideoJson() {
  videoJson.value = prettyJson(videoDefault)
  videoJsonError.value = ''
}

async function handleSave() {
  saving.value = true
  captionJsonError.value = ''
  proseJsonError.value = ''
  referenceJsonError.value = ''
  videoJsonError.value = ''

  let parsedCaption: Record<string, any>
  let parsedProse: Record<string, any>
  let parsedReference: Record<string, any>
  let parsedVideo: Record<string, any>

  try {
    parsedCaption = JSON.parse(captionJson.value)
  } catch (e: any) {
    captionJsonError.value = 'Caption JSON 解析失败: ' + e.message
    activeTab.value = 'caption'
    saving.value = false
    return
  }

  if (!parsedCaption.base_url || typeof parsedCaption.base_url !== 'string') {
    captionJsonError.value = 'Caption 校验失败: 缺少必填字段 base_url'
    activeTab.value = 'caption'
    saving.value = false
    return
  }
  if (parsedCaption.api_key === undefined || parsedCaption.api_key === null) {
    captionJsonError.value = 'Caption 校验失败: 缺少必填字段 api_key'
    activeTab.value = 'caption'
    saving.value = false
    return
  }
  if (!parsedCaption.model || typeof parsedCaption.model !== 'string') {
    captionJsonError.value = 'Caption 校验失败: 缺少必填字段 model'
    activeTab.value = 'caption'
    saving.value = false
    return
  }
  if (!parsedCaption.prompt_template || typeof parsedCaption.prompt_template !== 'string') {
    captionJsonError.value = 'Caption 校验失败: 缺少必填字段 prompt_template'
    activeTab.value = 'caption'
    saving.value = false
    return
  }

  try {
    parsedProse = JSON.parse(proseJson.value)
  } catch (e: any) {
    proseJsonError.value = 'Prose JSON 解析失败: ' + e.message
    activeTab.value = 'prose'
    saving.value = false
    return
  }

  if (!parsedProse.base_url || typeof parsedProse.base_url !== 'string') {
    proseJsonError.value = 'Prose 校验失败: 缺少必填字段 base_url'
    activeTab.value = 'prose'
    saving.value = false
    return
  }
  if (parsedProse.api_key === undefined || parsedProse.api_key === null) {
    proseJsonError.value = 'Prose 校验失败: 缺少必填字段 api_key'
    activeTab.value = 'prose'
    saving.value = false
    return
  }
  if (!parsedProse.model || typeof parsedProse.model !== 'string') {
    proseJsonError.value = 'Prose 校验失败: 缺少必填字段 model'
    activeTab.value = 'prose'
    saving.value = false
    return
  }
  if (!parsedProse.prompt_template || typeof parsedProse.prompt_template !== 'string') {
    proseJsonError.value = 'Prose 校验失败: 缺少必填字段 prompt_template'
    activeTab.value = 'prose'
    saving.value = false
    return
  }
  if (!parsedProse.prompt_template.includes('{prompt}')) {
    proseJsonError.value = 'Prose 校验失败: prompt_template 必须包含 {prompt} 占位符'
    activeTab.value = 'prose'
    saving.value = false
    return
  }

  try {
    parsedReference = JSON.parse(referenceJson.value)
  } catch (e: any) {
    referenceJsonError.value = '参考图 JSON 解析失败: ' + e.message
    activeTab.value = 'reference'
    saving.value = false
    return
  }

  if (parsedReference.enabled === undefined || parsedReference.enabled === null) {
    referenceJsonError.value = '参考图 校验失败: 缺少必填字段 enabled'
    activeTab.value = 'reference'
    saving.value = false
    return
  }

  try {
    parsedVideo = JSON.parse(videoJson.value)
  } catch (e: any) {
    videoJsonError.value = '视频 JSON 解析失败: ' + e.message
    activeTab.value = 'video'
    saving.value = false
    return
  }

  if (parsedVideo.enabled === undefined || parsedVideo.enabled === null) {
    videoJsonError.value = '视频 校验失败: 缺少必填字段 enabled'
    activeTab.value = 'video'
    saving.value = false
    return
  }

  if (parsedVideo.submit && parsedVideo.submit.url === undefined) {
    videoJsonError.value = '视频 校验失败: submit 配置中缺少 url 字段'
    activeTab.value = 'video'
    saving.value = false
    return
  }

  if (parsedVideo.poll && parsedVideo.poll.url_template === undefined) {
    videoJsonError.value = '视频 校验失败: poll 配置中缺少 url_template 字段'
    activeTab.value = 'video'
    saving.value = false
    return
  }

  config.value.ocr.format_code = ocrFormatCode.value
  config.value.caption = parsedCaption
  config.value.prose = parsedProse
  config.value.reference = parsedReference
  config.value.video = parsedVideo

  try {
    await configApi.update(config.value)
    ElMessage.success('已生效')

    if (location.search.includes('project_id')) {
      try {
        await projectApi.applyMagiMode()
      } catch (e) {
        console.warn('Apply magi mode failed', e)
      }
    }
    setTimeout(() => {
      router.push('/')
    }, 1500)
  } catch (e: any) {
    ElMessage.error('保存失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    saving.value = false
  }
}

async function handleReset() {
  if (!confirm('确定要恢复所有默认设置吗？')) return
  saving.value = true
  try {
    const res = await configApi.reset()
    if (res.data.success && res.data.config) {
      Object.assign(config.value, res.data.config)
    }
    syncEditors()
    captionJsonError.value = ''
    proseJsonError.value = ''
    referenceJsonError.value = ''
    videoJsonError.value = ''
    ElMessage.success('已恢复默认设置')
  } catch (e: any) {
    ElMessage.error('重置失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    saving.value = false
  }
}

async function onMagiModeChange() {
  if (location.search.includes('project_id')) {
    try {
      await projectApi.applyMagiMode()
    } catch (e) {
      console.warn('Apply magi mode failed', e)
    }
  }
}

onMounted(() => loadConfig())
</script>

<style scoped>
.settings-page {
  max-width: 1080px;
  margin: 0 auto;
}

.page-header h2 {
  margin: 0 0 20px 0;
  font-size: 24px;
}

.settings-card {
  margin-bottom: 20px;
}

.tab-content {
  padding: 8px 0;
}

.help-text {
  margin-top: 8px;
  font-size: 13px;
  color: #909399;
}

.help-text p {
  margin: 4px 0;
}

.help-text code {
  background: #f5f5f5;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}

.code-editor-wrapper {
  position: relative;
}

.form-item-block {
  width: 100%;
}

.code-editor {
  width: 100%;
  min-height: 200px;
  padding: 14px 16px;
  border: 1px solid #dcdfe6;
  border-radius: 6px;
  font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.55;
  color: #303133;
  background: #fafafa;
  resize: vertical;
  outline: none;
  box-sizing: border-box;
  tab-size: 4;
}

.code-editor:focus {
  border-color: #409eff;
  background: #f5f7fa;
}

.reset-btn {
  position: absolute;
  top: 8px;
  right: 12px;
}

.actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding-top: 16px;
  border-top: 1px solid #ebeef5;
}
</style>