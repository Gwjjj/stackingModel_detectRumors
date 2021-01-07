package com.gwjjj.weiborumor.Business;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.gwjjj.weiborumor.Domain.Comment;
import com.gwjjj.weiborumor.Domain.CommentReturn;
import com.gwjjj.weiborumor.Domain.Weibo;
import com.gwjjj.weiborumor.Utils.PythonApi;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Controller
@RequestMapping("/weibo")
public class WeiboBusiness {
    @Value("${weibo.clientId}")
    private String clientId;
    @Value("${weibo.clientSecret}")
    private String clientSecret; //  应用密码
    @Value("${weibo.redirectUri}")
    private String redirectUri;  //  回调页面
    Logger logger = LoggerFactory.getLogger(Logger.class);

    @ResponseBody
    @RequestMapping("/weiboInfo")
    private String getUserInfo(@RequestBody JSONObject jsonParam) throws JsonProcessingException {
        String access_token = (String) jsonParam.get("access_token");
        String since_id = "0";   // 若指定此参数，则返回ID比since_id大的微博（即比since_id时间晚的微博），默认为0。
        String max_id = "0";     // 若指定此参数，则返回ID小于或等于max_id的微博，默认为0。
        String count = "5";      // 单页返回的记录条数，最大不超过100，默认为20。
        String page = "1";       // 返回结果的页码，默认为1
        String base_app	 = "0";  // 是否只获取当前应用的数据。0为否（所有数据），1为是（仅当前应用），默认为0。
        String feature	 = "0";  // 过滤类型ID，0：全部、1：原创、2：图片、3：视频、4：音乐，默认为0。
        String trim_user	 = "0";  // 返回值中user字段开关，0：返回完整user字段、1：user字段仅返回user_id，默认为0。
        String weiboInfoUrl="https://api.weibo.com/2/statuses/home_timeline.json?access_token="+access_token
                + "&count=" + count + "&page=" + page;
        logger.info("weiboInfo =================  weibo api url: " + weiboInfoUrl);
        RestTemplate restTemplate = new RestTemplate();
        String forObject = restTemplate.getForObject(weiboInfoUrl, String.class);
        ObjectMapper objectMapper = new ObjectMapper();
        Map map = objectMapper.readValue(forObject, Map.class);
        List weiboList = (List) map.get("statuses");
        List<Weibo> reList = new ArrayList();
        for (int i = 0; i < weiboList.size(); i++) {
            Weibo weibo = new Weibo();
            Map weiboMap = (Map)weiboList.get(i);
            weibo.setUserName((String)((Map)weiboMap.get("user")).get("name"));
            weibo.setCreateTime((String)weiboMap.get("created_at"));
            weibo.setText((String)weiboMap.get("text"));
            weibo.setComment(String.valueOf(weiboMap.get("comments_count")));
            weibo.setZan(String.valueOf(weiboMap.get("attitudes_count")));
            weibo.setUserPicUrl((String)((Map)weiboMap.get("user")).get("profile_image_url"));
            weibo.setId(String.valueOf(weiboMap.get("id")));
            reList.add(weibo);
        }
        return new Gson().toJson(reList);
    }

    @ResponseBody
    @RequestMapping("/commentInfo")
    private String getCommentInfo(@RequestBody JSONObject jsonParam) throws JsonProcessingException {
        String accessToken = (String) jsonParam.get("access_token");
        String weiboId = (String) jsonParam.get("weibo_id");
        String count = "50";      // 单页返回的记录条数，最大不超过100，默认为50。
        String page = "1";       // 返回结果的页码，默认为1
        String commentInfoUrl="https://api.weibo.com/2/comments/show.json?access_token="+accessToken
                + "&id=" + weiboId + "&count=" + count + "&page=" + page;
        logger.info("commentInfo =================  weibo api url: " + commentInfoUrl);
        RestTemplate restTemplate = new RestTemplate();
        String forObject = restTemplate.getForObject(commentInfoUrl, String.class);
        ObjectMapper objectMapper = new ObjectMapper();
        Map map = objectMapper.readValue(forObject, Map.class);
        List commentList = (List) map.get("comments");
        List<Comment> reList = new ArrayList();
        String orgin_text = (String)((Map)map.get("status")).get("text");
        List<String> analyseText = new ArrayList<>();
        analyseText.add(orgin_text);
        for (int i = 0; i < commentList.size(); i++) {
            Comment comment = new Comment();
            Map commentMap = (Map)commentList.get(i);
            comment.setUserName((String)((Map)commentMap.get("user")).get("screen_name"));
            String text = (String)commentMap.get("reply_original_text");
            if(text == null){
                text = (String)commentMap.get("text");
            }
            analyseText.add(text);
            comment.setText(text);
            comment.setUserImg((String)((Map)commentMap.get("user")).get("profile_image_url"));
            reList.add(comment);
        }
        CommentReturn commentReturn = new CommentReturn();
        commentReturn.setComments(reList);
        commentReturn.setProb(PythonApi.doPython(analyseText));
        return new Gson().toJson(commentReturn);
    }
}
